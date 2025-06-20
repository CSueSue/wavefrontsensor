/*
 * This Arduino sketch controls a stepper motor using a DM542T driver, an LCD
 * display, a membrane keypad, and a rotary encoder. The user can set the
 * rotation angle and speed of the stepper motor using the keypad. The current
 * angle and speed are displayed on the LCD. The rotary encoder is used to
 * fine-tune the angle.
 */




// Rotary encoder setup
const int encoderPinA = 3;
const int encoderPinB = 2;
volatile int encoderValue = 0;

// Stepper motor driver pins
const int stepPin = 7;
const int dirPin = 8;
const int enablePin = 9;

// Variables
int targetAngle = 0;
int targetSpeed = 500; // Default speed in microseconds per step

void setup() {


  // Initialize rotary encoder
  pinMode(encoderPinA, INPUT);
  pinMode(encoderPinB, INPUT);
  attachInterrupt(digitalPinToInterrupt(encoderPinA), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoderPinB), updateEncoder, CHANGE);

  // Initialize stepper motor driver
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enablePin, OUTPUT);
  digitalWrite(enablePin, LOW); // Enable the driver
}

void loop() {

}


}

void updateEncoder() {
  int stateA = digitalRead(encoderPinA);
  int stateB = digitalRead(encoderPinB);
  if (stateA == stateB) {
    encoderValue++;
  } else {
    encoderValue--;
  }
  targetAngle += encoderValue;
  encoderValue = 0;
}

void moveStepper(int angle, int speed) {
  int steps = angleToSteps(angle);
  digitalWrite(dirPin, steps > 0 ? HIGH : LOW);
  steps = abs(steps);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(speed);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(speed);
  }
}

int angleToSteps(int angle) {
  // Assuming 200 steps per revolution and 1.8 degrees per step
  return (angle * 200) / 360;
}
