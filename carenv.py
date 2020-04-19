import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.markers as mmarkers
import torchvision.transforms as T
import imgutils

from PIL import Image
from kivy.vector import Vector
from collections import Counter

class CarEnv(object):
    def __init__(self, filename):
        self.filename = filename
        img = Image.open(self.filename).convert('L')
        self.sand = np.asarray(img)/255
        self.sand = self.sand.astype(int)
        self.max_y, self.max_x = self.sand.shape
        self.pos = Vector(int(self.max_x/2), int(self.max_y/2))
        self.angle = Vector(0,10).angle(self.pos)
        self.velocity = Vector(6, 0)
        self.wall_padding = 20
        self.rel_pos = Vector(0,0)
        self.max_angle = 10
        self.max_action = [self.max_angle]
        self.crop_size = 100
        self.goal_iter = 0
        self.goals = [Vector(1890, 150), Vector(140, 380)]
        self.last_distance = 0
        self.state_dim = (32, 32)
        self.action_dim = (1,)
        self._max_episode_steps = 4000
        # track rewards distribution
        self.rewards_distribution = Counter()

    def seed(self, seed):
        pass

    def reset(self):
        self.angle = np.random.randint(low=0, high=360)
        onsand = True
        while onsand:
          self.pos.x = np.random.randint(low=self.wall_padding, high=self.max_x-self.wall_padding)
          self.pos.y = np.random.randint(low=self.wall_padding, high=self.max_y-self.wall_padding)
          if self.sand[int(self.pos.y),int(self.pos.x)] <= 0:
              onsand = False
        self.velocity = Vector(2, 0).rotate(self.angle)
        return self.get_state()

    def random_action(self):
        rotation = np.random.randint(low=-self.max_angle, high=self.max_angle)
        return (rotation,)

    def step(self, action):
        rotation = action[0]
        self.angle += rotation
        self.pos = Vector(*self.velocity) + self.pos
        self.pos.x = int(self.pos.x)
        self.pos.y = int(self.pos.y)
        reward = 0
        done = False
        current_goal = self.goals[self.goal_iter]

        distance = self.pos.distance(current_goal)
        if self.sand[int(self.pos.y),int(self.pos.x)] > 0:
            self.velocity = Vector(0.5, 0).rotate(self.angle)
            reward = -1
            tag = "sand (-1)"
        else: # otherwise
            self.velocity = Vector(2, 0).rotate(self.angle)
            reward = -0.1
            tag = "road (-0.1)"

            if distance < self.last_distance:
              reward = 1
              tag = "road (+1)"

        if self.pos.x < self.wall_padding:
            self.pos.x = self.wall_padding
            reward = -5
            tag = "wall (-5)"
            done = True
        if self.pos.x > self.max_x - self.wall_padding:
            self.pos.x = self.max_x - self.wall_padding
            reward = -5
            tag = "wall (-5)"
            done = True
        if self.pos.y < self.wall_padding:
            self.pos.y = self.wall_padding
            reward = -5
            tag = "wall (-5)"
            done = True
        if self.pos.y > self.max_y - self.wall_padding:
            self.pos.y = self.max_y - self.wall_padding
            reward = -5
            tag = "wall (-5)"
            done = True

        if distance < 25:
            self.goal_iter = (self.goal_iter + 1) % len(self.goals)
            goal = self.goals[self.goal_iter]
            reward = 10
            tag = "goal (+10)"
            done = True

        self.last_distance = distance
        self.rewards_distribution[tag] += 1
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
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(self.state_dim[0], interpolation=Image.CUBIC),
                            T.ToTensor()])

        img = Image.open(self.filename).convert('L')

        # If we directly crop and rotate the image, we may loose information
        # from the edges. Hence we do the following:
        #   * Crop a larger portion of image
        #   * Rotate it to make the cropped image in the direction
        #     of car's orientation
        #   * Then crop it to required size
        crop_img = imgutils.center_crop_img(img, self.pos.x, self.pos.y, self.crop_size*3)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return screen.to(device)
