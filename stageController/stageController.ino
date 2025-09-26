const int limitPinY = 11;   // limit input
const int stepPinY  = 5;
const int dirPinY   = 7;
const int limitPinX = 9;   // limit input
const int stepPinX  = 4;
const int dirPinX   = 6;
const int pulsPerMM = 200; 
char packetBuffer[101];
int count = 0;

void setup() {
  pinMode(limitPinY, INPUT_PULLUP);
  pinMode(stepPinY, OUTPUT);
  pinMode(dirPinY, OUTPUT);
  pinMode(limitPinX, INPUT_PULLUP);
  pinMode(stepPinX, OUTPUT);
  pinMode(dirPinX, OUTPUT);
  Serial.begin(115200);

  
}



void homeStageY() {
  //Serial.println("Homing...");
  digitalWrite(dirPinY, LOW); // move toward switch
  while (digitalRead(limitPinY) == HIGH) {
    stepOnceY();
    //delayMicroseconds(1500);
  }
  Serial.println("Home found!");
}

void stepOnceY() {
  digitalWrite(stepPinY, HIGH);
  delayMicroseconds(800);
  digitalWrite(stepPinY, LOW);
  delayMicroseconds(800);
}

void homeStageX() {
  //Serial.println("Homing...");
  digitalWrite(dirPinX, LOW); // move toward switch
  while (digitalRead(limitPinX) == HIGH) {
    stepOnceX();
    //delayMicroseconds(1500);
  }
  Serial.println("Home found!");
}

void stepOnceX() {
  digitalWrite(stepPinX, HIGH);
  delayMicroseconds(800);
  digitalWrite(stepPinX, LOW);
  delayMicroseconds(800);
}


void move(double dx, double dy, int directionx, int directiony, double speed) {
  /// y in mm, direction 1 or 0 , speed in mm/s
  digitalWrite(dirPinY, directiony);
  digitalWrite(dirPinX, directionx);
  int delay = int(1/(pulsPerMM*2*speed)*1e6); //[micros]
  int NpulsesX = int(dx*pulsPerMM);
  int NpulsesY = int(dy*pulsPerMM);
  // Serial.print(delay);
  // Serial.print(",");
  // Serial.print(NpulsesX);
  // Serial.print(",");
  // Serial.print(NpulsesY);
  // Serial.print("\n");



  for (int i=0; i<NpulsesX; i++){
    digitalWrite(stepPinX, HIGH);
    delayMicroseconds(delay);
    digitalWrite(stepPinX, LOW);
    delayMicroseconds(delay);
    // if (digitalRead(limitPinX) == LOW ) {
    //   Serial.println("LIMIT HIT X! STOPPING");
    //   break;
    // }
  }

  for (int i=0; i<NpulsesY; i++){
    digitalWrite(stepPinY, HIGH);
    delayMicroseconds(delay);
    digitalWrite(stepPinY, LOW);
    delayMicroseconds(delay);
    // if (digitalRead(limitPinY) == LOW) {
    //   Serial.println("LIMIT HIT Y! STOPPING");
    //   break;
    // }
  }



}

void loop() {

  // while (true) {
  //     if (digitalRead(limitPinY) == LOW) {
  //         Serial.println("LIMIT HIT! STOPPING");
  //     }
  //     delayMicroseconds(1000000);
  // }
  bool stoploop = false;
  char value;

  while (count<100 and (not stoploop)){
     if (Serial.available()){
      value = Serial.read();
      //Serial.write(packetBuffer[count-1]);
      if (value== '\n'){stoploop = true;}
      else {packetBuffer[count]=value; count = count+1;}
     }

    }
  if (count>0 and stoploop == true){
    
    packetBuffer[count]=0;
    count = 0;  
    stoploop = false;
    char* command = strtok(packetBuffer, "&"); 
    while (command!=0){

      if (atoi(command)== 1){
        // move command.

        // read input values from serial command.
        command = strtok(0, "&");
        double dx = atof(command);
        command = strtok(0, "&");
        double dy = atof(command);
        command = strtok(0, "&");
        int dirx = atoi(command);
        command = strtok(0, "&");
        int diry = atoi(command);
        command = strtok(0, "&");
        double speed = atof(command);
        // Serial.print(dx);
        // Serial.print(",");
        // Serial.print(dy);
        // Serial.print(",");
        // Serial.print(dirx);
        // Serial.print(",");
        // Serial.print(diry);
        // Serial.print(",");
        // Serial.print(speed);
        // Serial.print("\n");
        move(dx,dy, dirx, diry,  speed);
        Serial.println("move done");
      }

      else if (atoi(command) == 2){
        homeStageX();
      }

      else if (atoi(command) == 3){
        homeStageY();
      }
      else if (atoi(command) == 4){
        Serial.print(digitalRead(limitPinX) );
        Serial.print(",");
        Serial.print(digitalRead(limitPinY) );
        Serial.print("\n");
      }
      else if (atoi(command) == 5){
        for (int i=0; i<1000; i++){
          digitalWrite(stepPinX, HIGH);
          delayMicroseconds(1000);
          digitalWrite(stepPinX, LOW);
          delayMicroseconds(1000);
          // if (digitalRead(limitPinX) == LOW ) {
          //   Serial.println("LIMIT HIT X! STOPPING");
          //   break;
          // }
        }
      }
      else if (atoi(command) == 6){
        for (int i=0; i<1000; i++){
          digitalWrite(stepPinY, HIGH);
          delayMicroseconds(1000);
          digitalWrite(stepPinY, LOW);
          delayMicroseconds(1000);
        }
      }

      else {}

      command = strtok(0, "&");
    }
  }




}