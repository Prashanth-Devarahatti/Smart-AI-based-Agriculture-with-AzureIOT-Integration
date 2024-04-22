# Smart-AI-based-Agriculture-with-AzureIOT-Integration

This project implements a Smart Agriculture IoT system using a Raspberry Pi to monitor and control environmental parameters such as temperature, humidity, soil moisture, and pH level. It utilizes fuzzy logic control and machine learning models for decision-making and communicates sensor data to an Azure IoT Hub.

## Dependencies:

- Python 3.x
- Adafruit_DHT (`Adafruit_DHT`)
- RPi.GPIO (`RPi.GPIO`)
- minimalmodbus (`minimalmodbus`)
- Azure IoT Device SDK for Python (`azure.iot.device`)
- NumPy (`numpy`)
- scikit-fuzzy (`skfuzzy`)
- joblib (`joblib`)

Make sure to install the required Python libraries in your Raspberry Pi environment before running the code.

## Hardware Setup:

1. Connect the DHT11 sensor to the GPIO pin 4 for temperature and humidity sensing.
2. Connect the soil moisture sensor to the GPIO pin 17 for soil moisture sensing.
3. Connect the pH sensor to the appropriate serial port (e.g., `/dev/ttyUSB0`) for pH level sensing.
4. Connect the water pump to GPIO pin 18 and the pesticide spray to GPIO pin 23 for irrigation and spraying control.

## Software Setup:

1. Initialize the Azure IoT Hub client with the provided connection string.
2. Define fuzzy control variables for temperature, humidity, soil moisture, pH level, water pump activation, and pesticide spray activation.
3. Define fuzzy membership functions and rules for watering and spraying control systems based on environmental parameters.
4. Load pre-trained machine learning models for predicting soil moisture and plant health.
5. Implement functions to read sensor data from the DHT11 sensor, soil moisture sensor, and pH sensor.
6. Apply fuzzy logic control and machine learning predictions to determine watering and spraying actions.
7. Send sensor data messages to the Azure IoT Hub using the IoT Hub client.

## Usage:

1. Run the Python script (`main.py`) on your Raspberry Pi to start the Smart Agriculture IoT system.
2. Monitor the environmental parameters and control actions through the Azure IoT Hub interface.
3. Customize fuzzy logic rules and machine learning models according to your specific agricultural requirements.
4. Adjust GPIO pin configurations and hardware connections as needed for your setup.

## Notes:

- Ensure that you have correctly wired and configured the sensors and actuators with the Raspberry Pi.
- Update the Azure IoT Hub connection string with your own IoT Hub information.
- Fine-tune fuzzy logic membership functions and rules to optimize irrigation and spraying decisions.
- Train and validate machine learning models using relevant agricultural data for accurate predictions.

## Contributors:

- Prashanth Devarahatti
---
Feel free to customize the README further with additional sections or details as needed!
