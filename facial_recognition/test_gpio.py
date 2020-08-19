import Jetson.GPIO as GPIO
import time

# Pin Definitons:
buzz_pin = 32  # Board pin 40
freq = 50
def main():
    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(buzz_pin, GPIO.OUT)  # set buzzer as output
    # Initial state for LEDs:
    GPIO.output(buzz_pin, GPIO.LOW)
    pwm = GPIO.PWM(buzz_pin, freq)
    
    while True:
        pwm.ChangeDutyCycle(10)
#    try:
#        
#        while True:
##            print("Waiting for button event")
##            GPIO.wait_for_edge(but_pin, GPIO.FALLING)
#            print("Beep!")
#            
##            GPIO.output(buzz_pin, GPIO.HIGH)
##            time.sleep(0.5)
##            GPIO.output(buzz_pin, GPIO.LOW)
##            time.sleep(0.1)
##            
#    finally:
#        GPIO.cleanup()  # cleanup all GPIOs
    
    

if __name__ == '__main__':
    main()