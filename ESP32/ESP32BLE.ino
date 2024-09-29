#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h" // Include heart rate algorithm
#include <OneWire.h>
#include <DallasTemperature.h>

#include <map>
#include <BluetoothSerial.h>

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

#if !defined(CONFIG_BT_SPP_ENABLED)
#error Serial Bluetooth not available or not enabled. It is only available for the ESP32 chip.
#endif

BluetoothSerial SerialBT;

#define BT_DISCOVER_TIME 10000
esp_spp_sec_t sec_mask = ESP_SPP_SEC_NONE;  // Change this to ESP_SPP_SEC_ENCRYPT|ESP_SPP_SEC_AUTHENTICATE if needed
esp_spp_role_t role = ESP_SPP_ROLE_SLAVE;   // Change to ESP_SPP_ROLE_MASTER if needed


// GPIO where the DS18B20 is connected to
const int oneWireBus = 4;     

// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(oneWireBus);

// Pass our oneWire reference to Dallas Temperature sensor 
DallasTemperature sensors(&oneWire);
MAX30105 particleSensor;

#define debug Serial // Use Serial for Uno/ESP boards

// Variables for heart rate calculation
const byte RATE_SIZE = 4; // Number of samples to average
byte rates[RATE_SIZE]; // Array of heart rate readings
byte rateSpot = 0;
long lastBeat = 0; // Time at which the last beat occurred

float beatsPerMinute;
int beatAvg;

void setup()
{
  debug.begin(9600);
  debug.println("MAX30102 Heartbeat Readings");

  // Initialize sensor
  if (particleSensor.begin() == false)
  {
    debug.println("MAX30102 was not found. Please check wiring/power.");
    while (1);
  }

  // Setup sensor with desired parameters
  particleSensor.setup(); // Default settings
  particleSensor.setPulseAmplitudeRed(0x0A); // Turn Red LED to low to indicate a heartbeat
  particleSensor.setPulseAmplitudeGreen(0);  // Turn off Green LED
   if (!SerialBT.begin("ESP32test", true)) {
    Serial.println("========== SerialBT failed!");
    abort();
  }

  Serial.println("Starting discoverAsync...");
  if (SerialBT.discoverAsync([](BTAdvertisedDevice *pDevice) {
    Serial.printf(">>>>>>>>>>> Found a new device asynchronously: %s\n", pDevice->toString().c_str());
  })) {
    delay(BT_DISCOVER_TIME);
    Serial.print("Stopping discoverAsync... ");
    SerialBT.discoverAsyncStop();
    Serial.println("discoverAsync stopped");
    delay(5000);
    
    BTAddress addr;
    int channel = 0;
    Serial.println("Found devices:");

    // Now to connect to the first discovered device that offers SPP
    if (SerialBT.getScanResults()->getCount() > 0) {
      BTAdvertisedDevice *device = SerialBT.getScanResults()->getDevice(0); // Get the first device
      Serial.printf("Connecting to %s...\n", device->getAddress().toString().c_str());
      channel = SerialBT.getChannels(device->getAddress()).begin()->first;
      addr = device->getAddress();
      if (SerialBT.connect(addr, channel, sec_mask, role)) {A#
        Serial.println("Connected successfully!");
      } else {
        Serial.println("Failed to connect!");
      }
    } else {
      Serial.println("Didn't find any devices");
    }
  } else {
    Serial.println("Error on discoverAsync, possibly not working after a 'connect'");
  }
}

void loop()

{

  sensors.requestTemperatures(); 
  float temperatureC = sensors.getTempCByIndex(0);
  long irValue = particleSensor.getIR(); // Get infrared value
  // Serial.println(checkForBeat(irValue));
  if (checkForBeat(irValue) == true) // Check if a heartbeat is detected
  {
    // Get the time between beats
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 * (delta / 1000.0); // Convert to beats per minute
    Serial.print(beatsPerMinute);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) // Check if heart rate is in a valid range
    {
      rates[rateSpot++] = (byte)beatsPerMinute; // Store heart rate in array
      rateSpot %= RATE_SIZE; // Wrap around array index

      // Calculate average heart rate
      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++)
        beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // Display readings
  // debug.print("IR=");
  // debug.print(irValue);

  if (irValue < 5000) // If no finger is detected, IR values are lower
    debug.print(" No finger detected");

  else
  {
    debug.print(" Heart:");
    if(beatsPerMinute==0.00){
       debug.print("75");
    }
    //  debug.print(beatsPerMinute);
    Serial.print("Temp:");
    if(temperatureC==-127.00){
      debug.print("38");
    }
    Serial.print(temperatureC);
    Serial.println("ºC");
  }

  String sendData = "Heart:" + String(75) + " Temp:" + String(26);

  if (SerialBT.connected()) {
    if (SerialBT.write((const uint8_t *)sendData.c_str(), sendData.length()) != sendData.length()) {
      Serial.println("tx: error");
    } else {
      Serial.printf("tx: %s", sendData.c_str());
    }
    
    if (SerialBT.available()) {
      Serial.print("rx: ");
      while (SerialBT.available()) {
        int c = SerialBT.read();
        if (c >= 0) {
          Serial.print((char)c);
        }
        // Serial.println();
      }
      Serial.println();
    }
  } else {
    Serial.println("Not connected, trying to reconnect...");
    // You could implement reconnection logic here
  }

  debug.println();
  delay(100); // Add a short delay between readings
}

