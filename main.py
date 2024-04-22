import time
import Adafruit_DHT
import RPi.GPIO as GPIO
import minimalmodbus
from azure.iot.device import IoTHubDeviceClient, Message
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.externals import joblib

CONNECTION_STRING = "your-iot-hub-connection-string"

# Define GPIO pins
WATER_PUMP_PIN = 18
PESTICIDE_SPRAY_PIN = 23

# Define fuzzy control variables
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
soil_moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'soil_moisture')
ph_level = ctrl.Antecedent(np.arange(0, 15, 0.1), 'ph_level')
water_pump = ctrl.Consequent(np.arange(0, 101, 1), 'water_pump')
pesticide_spray = ctrl.Consequent(np.arange(0, 101, 1), 'pesticide_spray')

# Define fuzzy membership functions and rules for watering and spraying control systems
# Example fuzzy membership functions and rules:
temperature['low'] = fuzz.trimf(temperature.universe, [0, 20, 40])
temperature['medium'] = fuzz.trimf(temperature.universe, [30, 50, 70])
temperature['high'] = fuzz.trimf(temperature.universe, [60, 80, 100])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 20, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 80, 100])

soil_moisture['dry'] = fuzz.trimf(soil_moisture.universe, [0, 20, 40])
soil_moisture['moist'] = fuzz.trimf(soil_moisture.universe, [30, 50, 70])
soil_moisture['wet'] = fuzz.trimf(soil_moisture.universe, [60, 80, 100])

ph_level['acidic'] = fuzz.trimf(ph_level.universe, [0, 3, 6])
ph_level['neutral'] = fuzz.trimf(ph_level.universe, [5, 7, 9])
ph_level['alkaline'] = fuzz.trimf(ph_level.universe, [8, 11, 14])

water_pump['low'] = fuzz.trimf(water_pump.universe, [0, 30, 60])
water_pump['medium'] = fuzz.trimf(water_pump.universe, [40, 70, 100])
water_pump['high'] = fuzz.trimf(water_pump.universe, [80, 100, 100])

pesticide_spray['low'] = fuzz.trimf(pesticide_spray.universe, [0, 30, 60])
pesticide_spray['medium'] = fuzz.trimf(pesticide_spray.universe, [40, 70, 100])
pesticide_spray['high'] = fuzz.trimf(pesticide_spray.universe, [80, 100, 100])

rule1 = ctrl.Rule(temperature['high'] | humidity['high'] | soil_moisture['dry'], water_pump['high'])
rule2 = ctrl.Rule(ph_level['acidic'] | soil_moisture['dry'], pesticide_spray['high'])

watering_ctrl = ctrl.ControlSystem([rule1])
spraying_ctrl = ctrl.ControlSystem([rule2])

watering = ctrl.ControlSystemSimulation(watering_ctrl)
spraying = ctrl.ControlSystemSimulation(spraying_ctrl)

# Load pre-trained machine learning models
SOIL_MOISTURE_MODEL_PATH = "soil_moisture_model.pkl"
PLANT_HEALTH_MODEL_PATH = "plant_health_model.pkl"

def iothub_client_init():
    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    return client

def send_message(client, message):
    msg = Message(message)
    client.send_message(msg)
    print("Message sent:", message)

def read_dht_sensor_data():
    DHT_SENSOR = Adafruit_DHT.DHT11
    DHT_PIN = 4
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
    return humidity, temperature

def read_soil_moisture():
    SOIL_MOISTURE_PIN = 17
    GPIO.setup(SOIL_MOISTURE_PIN, GPIO.IN)
    soil_moisture = GPIO.input(SOIL_MOISTURE_PIN)
    return soil_moisture

def read_ph_level():
    instrument = minimalmodbus.Instrument('/dev/ttyUSB0', 1)
    instrument.serial.baudrate = 9600
    ph_level = instrument.read_float(0, functioncode=4)
    return ph_level

def predict_soil_moisture(humidity, temperature):
    model = joblib.load(SOIL_MOISTURE_MODEL_PATH)
    features = [[humidity, temperature]]
    predicted_soil_moisture = model.predict(features)
    return predicted_soil_moisture

def predict_plant_health(temperature, humidity, soil_moisture, ph_level):
    model = joblib.load(PLANT_HEALTH_MODEL_PATH)
    features = [[temperature, humidity, soil_moisture, ph_level]]
    predicted_plant_health = model.predict(features)
    return predicted_plant_health

if __name__ == "__main__":
    GPIO.setmode(GPIO.BCM)
    client = iothub_client_init()
    try:
        while True:
            humidity, temperature = read_dht_sensor_data()
            soil_moisture = read_soil_moisture()
            ph_level = read_ph_level()

            # Fuzzy logic control
            watering.input['temperature'] = temperature
            watering.input['humidity'] = humidity
            watering.input['soil_moisture'] = soil_moisture
            watering.compute()

            spraying.input['ph_level'] = ph_level
            spraying.input['soil_moisture'] = soil_moisture
            spraying.compute()

            water_pump_activation = watering.output['water_pump']
            pesticide_spray_activation = spraying.output['pesticide_spray']

            # Machine learning predictions
            predicted_soil_moisture = predict_soil_moisture(humidity, temperature)
            predicted_plant_health = predict_plant_health(temperature, humidity, soil_moisture, ph_level)

            # Control actions based on fuzzy logic and machine learning
            if water_pump_activation > 50:
                # Activate water pump
                GPIO.setup(WATER_PUMP_PIN, GPIO.OUT)
                GPIO.output(WATER_PUMP_PIN, GPIO.HIGH)
            else:
                # Deactivate water pump
                GPIO.setup(WATER_PUMP_PIN, GPIO.OUT)
                GPIO.output(WATER_PUMP_PIN, GPIO.LOW)

            if pesticide_spray_activation > 50:
                # Activate pesticide spray
                GPIO.setup(PESTICIDE_SPRAY_PIN, GPIO.OUT)
                GPIO.output(PESTICIDE_SPRAY_PIN, GPIO.HIGH)
            else:
                # Deactivate pesticide spray
                GPIO.setup(PESTICIDE_SPRAY_PIN, GPIO.OUT)
                GPIO.output(PESTICIDE_SPRAY_PIN, GPIO.LOW)

            sensor_data = {
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "ph_level": ph_level
            }

            message = {
                "sensor_data": sensor_data
            }

            send_message(client, message)
            time.sleep(5)
    except KeyboardInterrupt:
        print("IoTHubClient stopped")
    finally:
        GPIO.cleanup()
