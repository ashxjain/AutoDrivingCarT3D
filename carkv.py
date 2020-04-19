# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

import carenv
import t3d
import torch

Builder.load_string("""
#:kivy 1.0.9
# ref: https://kivy.org/docs/tutorials/pong.html

<Car>:
    size: 20, 10
    origin: 10, 5
    canvas:
        PushMatrix
        Rotate:
            angle: self.angle
            origin: self.center
        Rectangle:
            pos: self.pos
            size: self.size
            source: "./images/car.png"
        PopMatrix

<Game>:
    car: game_car
    canvas:
        Rectangle:
            pos: self.pos
            size: 1429, 660
            source: "./images/citymap.png"
    Car:
        id: game_car
        center: self.parent.center
""")

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '715') #, '1429')
Config.set('graphics', 'height', '330') #'660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Initializing the map
first_update = True
def init():
    global longueur,largeur, sand, first_update, policy, env
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    first_update = False

    env_name = "AutoDrivingCarModels"
    seed = 0
    file_name = "%s_%s_%s" % ("T3D", env_name, str(seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    env = carenv.CarEnv("./images/MASK1.png")
    max_episode_steps = env._max_episode_steps
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = env.state_dim
    action_dim = env.action_dim[0]
    max_action = env.max_action
    policy = t3d.T3D(state_dim, action_dim, max_action)
    policy.load(file_name, './pytorch_models/')
    env.reset()

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def init_state(self, new_pos, rotation):
        self.pos = new_pos
        self.rotation = rotation 
        self.angle = self.rotation

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global longueur
        global largeur
        global first_update, env, policy, sand
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.car.init_state(env.pos, env.angle)
        else:
            env.pos.x, env.pos.y = int(self.car.pos[0]), largeur - int(self.car.pos[1])
            env.angle = int(self.car.angle)
        obs = env.get_state()
        action = policy.select_action(obs)
        self.car.move(int(action[0]))
        if sand[int(env.pos.x),int(env.pos.y)] > 0:
            self.velocity = Vector(0.5, 0).rotate(env.angle)
        else:
            self.velocity = Vector(2, 0).rotate(env.angle)
        if self.car.x < 20:
            self.car.x = 20
        if self.car.x > longueur - 20:
            self.car.x = longueur - 20
        if self.car.y < 20:
            self.car.y = 20
        if self.car.y > largeur - 20:
            self.car.y = largeur - 20


# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
