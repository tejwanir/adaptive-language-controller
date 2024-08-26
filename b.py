import json
import numpy as np



def moveUp():
    pass
def moveDown():
    pass
def moveRight():
    pass
def moveLeft():
    pass

def solve():


    # Change from left to right (type 2->3)                                       right         
    # Change from right to left (type 4->1)                                       left         
    # Keep moving left (type 1->2)                                                left
    # Keep moving right (type 3->4)                                               right 
    # Move up ("Please move your hand up")                                        up        
    # Move down ("Please move your hand down")                                    down            
    # Moving wrong direction to right ("Move the other way, rightward")           right                                           
    # Moving wrong direction to left ("Move the other way, leftward")             down                                         


    while True:
        x = "Move your hand to the right" #input


        if x == "Move the other way, rightward":
            moveRight()
        elif x == "Move the other way, leftward":
            moveLeft()
        elif x == "Please move your hand up":
            moveUp()
        elif x == "Please move your hand down":
            moveDown()
        else:
            if "keep" in x["phrase"] and (" in" in x["phrase"] or "forward" in x["phrase"]):
                moveRight()
            elif ("far" in x["phrase"] or "keep" in x["phrase"]) and (" away" in x["phrase"] or " out" in x["phrase"]):
                moveLeft()
            elif "towards" in x["phrase"] and (" me" in x["phrase"] or " myself" in x["phrase"]):
                moveLeft()
            elif ("forward" in x["phrase"] or " away" in x["phrase"]) and " you" in x["phrase"]:
                moveLeft()
            elif "towards" in x["phrase"] and " you" in x["phrase"]:
                moveRight()
            elif ("further" in x["phrase"]):
                moveRight()
        


    






if __name__ == "__main__":
    solve()