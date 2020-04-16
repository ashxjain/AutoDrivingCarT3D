import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.markers as mmarkers
import torchvision.transforms as T

from PIL import Image
from kivy.vector import Vector

import imgutils

class CarEnv(object):
    def __init__(self, image_file):
        self.pil_img = Image.open(image_file).convert('L')
        self.sand = np.asarray(self.pil_img)/255
        self.sand = self.sand.astype(int)
        self.max_y, self.max_x = self.sand.shape
        self.pos = Vector(int(self.max_x/2), int(self.max_y/2))
        self.angle = Vector(0,10).angle(self.pos)
        self.rel_pos = Vector(0,0)
        self.max_angle = 10
        self.min_velocity = 0.5
        self.max_velocity = 10
        self.max_action = [self.max_velocity, self.max_angle]
        self.crop_size = 100
        self.goal_iter = 0
        self.goals = [Vector(1420, 38), Vector(9, 575)]
        self.last_distance = 0
        self.state_dim = (32, 32)
        self.action_dim = (2,)
        self._max_episode_steps = 2000
      
    def seed(self, seed):
        pass
    
    def reset(self):
        on_sand = True
        while on_sand:
          self.pos.x = np.random.randint(low=0, high=self.max_x-5)
          self.pos.y = np.random.randint(low=0, high=self.max_y-5)
          if self.sand[int(self.pos.y),int(self.pos.x)] <= 0:
            on_sand = False
        self.angle = Vector(0,10).angle(self.pos)
        return self.get_state()
    
    def random_action(self):
        vel = np.random.randint(low=self.min_velocity, high=self.max_velocity)
        ang = np.random.randint(low=-self.max_angle, high=self.max_angle)
        return (vel, ang)
    
    def step(self, action):
        vel, ang = action
        self.angle += ang
        reward = 0
        done = False
        current_goal = self.goals[self.goal_iter]
        distance = self.pos.distance(current_goal)
        if self.sand[int(self.pos.y),int(self.pos.x)] > 0:
            self.pos += Vector(vel, 0).rotate(self.angle)
            reward = -1
        else: # otherwise
            self.pos += Vector(vel, 0).rotate(self.angle)
            reward = -0.1
            if distance < self.last_distance:
                reward = 1
        if self.pos.x < 5:
            self.pos.x = 5
            reward = -0.5
            done = True
        if self.pos.x > self.max_x - 5:
            self.pos.x = self.max_x - 5
            reward = -0.5
        if self.pos.y < 5:
            self.pos.y = 5
            reward = -0.5
            done = True
        if self.pos.y > self.max_y - 5:
            self.pos.y = self.max_y - 5
            reward = -0.5
            done = True
        if distance < 25:
            self.goal_iter = (self.goal_iter + 1) % len(self.goals)
            goal = self.goals[self.goal_iter]
            reward = 1
            done = True
        self.last_distance = distance
        return self.get_state(), reward, done
    
    def render(self):
        # Create figure and axes
        fig, ax = plt.subplots(1, 5, figsize=(30, 6))

        # Display the image
        ax[0].imshow(self.sand, cmap='gray', vmin=0, vmax=1)
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (self.pos.x - int(self.crop_size/2), self.pos.y - int(self.crop_size/2)),
            self.crop_size, self.crop_size,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        # Add the patch to the Axes
        ax[0].add_patch(rect)
        ax[0].set_title("x=%d,y=%d,angle=%d" % (self.pos.x, self.pos.y, self.angle))
        
        marker = mmarkers.MarkerStyle(marker="$ \\rightarrow$")
        marker._transform = marker.get_transform().rotate_deg(self.angle)
        ax[0].scatter(self.pos.x, self.pos.y, s=50, c='red', marker=marker)
        self.get_state(ax).cpu().numpy()
        plt.show()
        
    def get_state(self, ax=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(self.state_dim[0], interpolation=Image.CUBIC),
                            T.ToTensor()])
        
        crop_img = imgutils.center_crop_img(self.pil_img, self.pos.x, self.pos.y, self.crop_size*3)
        if ax is not None:
          imgutils.show_img(ax[1], crop_img, "large crop")

        r_img = imgutils.rotate_img(crop_img, -self.angle)
        if ax is not None:
          imgutils.show_img(ax[2], r_img, "rotated crop")

        r_img_x, r_img_y = r_img.size
        crop_img = imgutils.center_crop_img(r_img, int(r_img_x/2), int(r_img_y/2), self.crop_size)
        if ax is not None:
          imgutils.show_img(ax[3], crop_img, "final crop")

        np_img = np.asarray(crop_img)/255
        np_img = np_img.astype(int)
        screen = np.ascontiguousarray(np_img, dtype=np.float32) 
        screen = torch.from_numpy(screen)
        screen = resize(screen)
        if ax is not None:
            np_img = screen.squeeze(0).numpy()
            np_img = np_img.astype(int)
            ax[4].imshow(np_img, cmap='gray', vmin=0, vmax=1)
            marker = mmarkers.MarkerStyle(marker="$ \\rightarrow$")
            ax[4].scatter(self.state_dim[0]/2, self.state_dim[1]/2, s=100, c='red', marker=marker)
            ax[4].set_title("final resized img")
        return screen.to(device)
