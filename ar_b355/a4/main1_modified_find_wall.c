#pragma config(Sensor, dgtl1,  front_left,     sensorTouch)
#pragma config(Sensor, dgtl2,  front_right,    sensorTouch)
#pragma config(Sensor, dgtl3,  right_front,    sensorTouch)
#pragma config(Sensor, dgtl4,  right_rear,     sensorTouch)
#pragma config(Motor,  port2,           leftMotor,     tmotorVex393, openLoop, reversed)
#pragma config(Motor,  port3,           rightMotor,    tmotorVex393, openLoop)
//*!!Code automatically generated by 'ROBOTC' configuration wizard               !!*//

//// Constants
int turn_speed = 50;
int soft_turn_speed = 30;
int forward_speed = 57;
int hyper_speed = 85;
float state_threshold = 30;

//// Variables
int prev_state;
int state;
int front_left_bump;
int front_right_bump;
int right_front_bump;
int right_rear_bump;
float state_counter;


bool sensor_contact_turn()
{
	front_left_bump = SensorValue(front_left);
  front_right_bump = SensorValue(front_right);
  right_front_bump = SensorValue(right_front);
  right_rear_bump = SensorValue(right_rear);

	if (front_left_bump == 1 || front_right_bump == 1 || right_front_bump == 1)
  {
  		return true;
  }
  else
  {
  		return false;
  }
}

bool sensor_contact()
{
	front_left_bump = SensorValue(front_left);
  front_right_bump = SensorValue(front_right);
  right_front_bump = SensorValue(right_front);
  right_rear_bump = SensorValue(right_rear);

	if (front_left_bump == 1 || front_right_bump == 1 || right_front_bump == 1 || right_rear_bump == 1)
  {
  		return true;
  }
  else
  {
  		return false;
  }
}

// Move for a given amount of time in direction left (-1), forward (0), right (1), backwards(2), soft left(3), or soft right(4)
void move(int dir, int time, int speed)
{
		switch(dir)
    {
    		case -1: // Left
            motor[leftMotor] = -speed;
        		motor[rightMotor] = speed;
            break;
        case 0: // Forwards
            motor[leftMotor] = speed;
        		motor[rightMotor] = speed;
            break;
        case 1: // Right
            motor[leftMotor] = speed;
        		motor[rightMotor] = -speed;
            break;
        case 2: // Backwards
            motor[leftMotor] = -speed;
        		motor[rightMotor] = -speed;
            break;
        case 3: // Soft Left
            motor[leftMotor] = 0;
        		motor[rightMotor] = speed;
            break;
        case 4: // Soft Right
            motor[leftMotor] = speed;
        		motor[rightMotor] = 0;
            break;
        default:
         		// Do nothing
         		break;
    }

    // Wait for passed time then stop
    wait1Msec(time);
    motor[leftMotor] = 0;
    motor[rightMotor] = 0;
}

// Turn and move forward in search for wall
void start_maze(int f_speed)
{
		if(prev_state == 1 || prev_state == 2)
		{
				// Turn right
		    move(1,400,turn_speed);
		    wait1Msec(100);
		    // Move forward
		    move(0,100,f_speed);
		    wait1Msec(100);
		}
		else
		{
				// Turn right
		    move(1,150,turn_speed);
		    wait1Msec(200);
		    // Move forward
		    move(0,1000,f_speed);
		    wait1Msec(200);
		}
}

// State 1. To be put in this state one right touch sensor must be activated
void side_wall_correct()
{
    // Right side contact
    if(right_front_bump == 1 || right_rear_bump == 1)
    {
        // Optimal - do nothing
        if (right_front_bump == 1 && right_rear_bump == 1)
        {
            return;
        }
        // Angled at wall - soft turn left
        if(right_front_bump == 1)
        {
            move(3,300,soft_turn_speed);
        }
        // Angled away from wall - soft turn right
        if(right_rear_bump == 1)
        {
        		int counter = 0;
        		while(!sensor_contact_turn() && counter < state_threshold)
        		{
        				move(4,200,turn_speed);
        				counter++;
        				if(counter >= state_threshold)
        				{
        						move(2,400,soft_turn_speed);
        				}
        		}
        }
    }
}

// State 3. To be put in this state one front touch sensor must be activated
void front_wall_correct()
{
		// Front sensor
    if (front_left_bump == 1 && front_right_bump == 1)
    {
        // Move backwards then turn left
		    move(2,300,turn_speed);
		    wait1Msec(100);
		    move(-1,650,turn_speed);
		    wait1Msec(100);
    }
    else if (front_right_bump == 1)
  	{
  			move(-1,650,soft_turn_speed);
  	}
  	else // Left front is tripped
  	{
  			if (right_front_bump == 1 || right_rear_bump == 1)
  			{
  					move(2,300,turn_speed);
				    wait1Msec(100);
				    move(-1,550,turn_speed); // CHANGED
				    wait1Msec(100);
  			}
  			else
  			{
  					move(1,300,soft_turn_speed);
  			}
  	}
}

// We're stuck in some state. Back up and turn left
void escape()
{
		move(2,600,turn_speed);
		move(-1,600,turn_speed);
}

// State 0. Turn and move forward in search for wall
void find_wall()
{
		int counter = 0;
		while(!sensor_contact() && counter < 7)
		{
				move(4,70,turn_speed);
				counter++;
		}
		move(0,150,forward_speed);
}

// Call the function affiliated with the passed state
void router(int state)
{
    switch(state)
    {
        case 0: // Find wall
        		find_wall();
            break;
        case 1: // Postion self on wall
            side_wall_correct();
            break;
        case 2: // Full speed ahead
            move(0,70,hyper_speed);
            break;
        case 3:
            front_wall_correct();
            break;
         default:
         		// TODO
         		break;
    }
}

/* Assignment 4
States:
0: Find wall
1: Wall correct
2: Full speed ahead
3: Hit forward wall
4:
*/

task main()
{
    wait1Msec(2000); // give stuff time to turn on

    state = 0;
    state_counter = 0;

    while(state == 0)
    {
        start_maze(forward_speed);

        // Get current state
      	front_left_bump = SensorValue(front_left);
		    front_right_bump = SensorValue(front_right);
		    right_front_bump = SensorValue(right_front);
		    right_rear_bump = SensorValue(right_rear);

        if (front_left_bump == 1 || front_right_bump == 1 || right_front_bump == 1 || right_rear_bump == 1)//TODO test
		    {
		        state = 1;
		    }
    }

    // Main control loop
    while(true)
    {
        writeDebugStreamLine("State: %d", state);
        writeDebugStreamLine("Prev state: %.6f", prev_state);
        writeDebugStreamLine("State counter: %.6f", state_counter);

        // Check if stuck in current state
		    if(prev_state == state)
		    {
		    		writeDebugStreamLine("should increment");
		    		state_counter++;
		    }
		    else
		    {
		    		state_counter = 0;
		    }

		    if(state_counter > state_threshold)
		    {
		    		if(state == 1)
		    		{
		    				if (right_front_bump == 1)
	        			{
	        					move(-1,200,turn_speed);
	        			}
	        			else
	        			{
	        					move(1,200,turn_speed);
	        			}
		    		}
		    		else if(state == 2)
		    		{
		    				move(-2,250,forward_speed);
		    				move(0,300,hyper_speed);
		    		}
		    		else
			    	{
				    		escape();
			    	}
			    	state_counter = 0;
		    }

        // Get current state
        front_left_bump = SensorValue(front_left);
		    front_right_bump = SensorValue(front_right);
		    right_front_bump = SensorValue(right_front);
		    right_rear_bump = SensorValue(right_rear);

		    // Any touch sensor activated
		    if (front_left_bump == 1 || front_right_bump == 1 || right_front_bump == 1 || right_rear_bump == 1)
		    {
		      // First check front sensors
		      if (front_left_bump == 1 || front_right_bump == 1)
			    {
			        prev_state = state;
		          state = 3;
		          state_counter += 5;
			    }
          // Then both side sensors
	        else if (right_front_bump == 1 && right_rear_bump == 1)
	        {
	            prev_state = state;
	            state = 2;
	            state_counter -= .85;
	        }
	        // Else wall correct
	        else
	        {
	            prev_state = state;
	            state = 1;
	        }
		    }
		    // Find wall if nothing is touching
		    else
		    {
		        prev_state = state;
		        state = 0;
		        state_counter += 3;
		    }

		    router(state);
    }
}
