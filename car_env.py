import numpy as np
import pyglet


pyglet.clock.set_fps_limit(100)


class CarEnv(object):
    n_sensor = 10
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (600, 600)
    sensor_max = 100.
    start_point = [300, 300]
    speed = 50.
    dt = 0.1

    def __init__(self, discrete_action=False, carn = 3):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]
        self.carn = carn
        self.terminal = []
        self.collision = []
        self.target = []
        self.distance = []
        self.distance_ = []
        self.car_info = []
        self.sensor_info = []
        for n in range(self.carn):
            self.terminal.append(False)
            self.collision.append(False)
            self.target.append(False)
            self.distance.append(0)
            self.distance_.append(0)
            self.car_info.append(np.array([0, 0, 0, 20, 40], dtype=np.float64))
            self.sensor_info.append(self.sensor_max + np.zeros((self.n_sensor, 3)))

        self.obstacle_coords = np.array([
            [500, 500],
            [600, 500],
            [600, 600],
            [500, 600],
        ])
        self.obstacle_coords1 = np.array([
            [100, 100],
            [150, 100],
            [150, 500],
            [100, 500],
        ])
        self.obstacle_coords2 = np.array([
            [450, 100],
            [500, 100],
            [500, 500],
            [450, 500],
        ])
        self.obstacle_coords3 = np.array([
            [250, 100],
            [350, 100],
            [350, 150],
            [250, 150],
        ])
        self.obstacle_coords4 = np.array([
            [250, 450],
            [350, 450],
            [350, 500],
            [250, 500],
        ])

    def step(self, action, cari):

        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[cari][2] += action * np.pi/18  # max r = 6 degree
        if self.car_info[cari][2] > np.pi:
            self.car_info[cari][2] = - (2*np.pi - self.car_info[cari][2])
        if self.car_info[cari][2] < -np.pi:
            self.car_info[cari][2] = (2*np.pi - abs(self.car_info[cari][2]))
        self.car_info[cari][:2] = self.car_info[cari][:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[cari][2]),
                                                             np.sin(self.car_info[cari][2])])

        self._update_sensor(cari)

        x1 = self.car_info[cari][0]
        y1 = self.car_info[cari][1]
        dx1 = abs(x1 - 550) / 600
        dy1 = abs(y1 - 550) / 600
        self.distance_[cari] = (dx1 ** 2 + dy1 ** 2) ** 0.5
        reward = (self.distance[cari] - self.distance_[cari])
        self.distance[cari] = self.distance_[cari]
        if 500 < self.car_info[cari][0] < 600 and 500 < self.car_info[cari][1] < 600:
            self.target[cari] = True
            reward = 10
        s = self._get_state(cari)
        if self.collision[cari]:
            reward = - 5
        if self.target[cari] or self.collision[cari]:
            self.terminal[cari] = True

        return s, reward, self.terminal[cari]

    def reset(self, cari):
        self.terminal[cari] = False
        self.collision[cari] = False
        self.target[cari] = False
        self.car_info[cari][:3] = np.array([*self.start_point, - np.pi / 2])
        self._update_sensor(cari)
        return self._get_state(cari)

    def render(self):

        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info,self.sensor_info,
                                  self.obstacle_coords, self.obstacle_coords1,
                                 self.obstacle_coords2, self.obstacle_coords3,
                                 self.obstacle_coords4, )

        self.viewer.render()

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a

    def set_fps(self, fps=300):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self, cari):
        s = self.sensor_info[cari][:, 0]/self.sensor_max
        angle = self.car_info[cari][2]
        x = self.car_info[cari][0]
        y = self.car_info[cari][1]
        dx = abs(x-550)/600
        dy = abs(y-550)/600
        d = (dx**2+dy**2)**0.5
        a = np.arctan(dy/dx)*np.pi
        zta = 0
        if x <= 550 and y <= 550:
            zta = a - angle
            if zta > np.pi:
                zta = -(2*np.pi-zta)
        if x >= 550 and y >= 550:
            a = -(np.pi - a)
            zta = a - angle
            if zta < - np.pi:
                zta = 2*np.pi + zta
        if x < 550 and y > 550:
            a = - a
            zta = a - angle
            if zta < - np.pi:
                zta = 2*np.pi + zta
        if x > 550 and y < 550:
            a = np.pi - a
            zta = a - angle
            if zta > np.pi:
                zta = -(2*np.pi - zta)

        s = np.hstack((s, angle/np.pi, d, zta/np.pi))
        return s

    def _update_sensor(self, cari):
        cx, cy, rotation = self.car_info[cari][:3]

        n_sensors = len(self.sensor_info[cari])
        sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        xs = cx + (np.zeros((n_sensors, ))+self.sensor_max) * np.cos(sensor_theta)
        ys = cy + (np.zeros((n_sensors, ))+self.sensor_max) * np.sin(sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])    # shape (5 sensors, 2)

        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[cari][:, -2:] = np.vstack([rotated_x+cx, rotated_y+cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info[cari])):
            s = self.sensor_info[cari][si, -2:] - q
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[cari][si, -2:]]

            # obstacle collision
            for oi1 in range(len(self.obstacle_coords1)):
                p = self.obstacle_coords1[oi1]
                r = self.obstacle_coords1[(oi1 + 1) % len(self.obstacle_coords1)] - self.obstacle_coords1[oi1]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi2 in range(len(self.obstacle_coords2)):
                p = self.obstacle_coords2[oi2]
                r = self.obstacle_coords2[(oi2 + 1) % len(self.obstacle_coords2)] - self.obstacle_coords2[oi2]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi3 in range(len(self.obstacle_coords3)):
                p = self.obstacle_coords3[oi3]
                r = self.obstacle_coords3[(oi3 + 1) % len(self.obstacle_coords3)] - self.obstacle_coords3[oi3]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            for oi4 in range(len(self.obstacle_coords4)):
                p = self.obstacle_coords4[oi4]
                r = self.obstacle_coords4[(oi4 + 1) % len(self.obstacle_coords4)] - self.obstacle_coords4[oi4]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = q + u * s
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(u*s))
            # window collision
            win_coord = np.array([
                [0, 0],
                [self.viewer_xy[0], 0],
                [*self.viewer_xy],
                [0, self.viewer_xy[1]],
                [0, 0],
            ])
            for oi in range(4):
                p = win_coord[oi]
                r = win_coord[(oi + 1) % len(win_coord)] - win_coord[oi]
                if np.cross(r, s) != 0:  # may collision
                    t = np.cross((q - p), s) / np.cross(r, s)
                    u = np.cross((q - p), r) / np.cross(r, s)
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection = p + t * r
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(np.linalg.norm(intersection - q))

            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[cari][si, 0] = distance
            self.sensor_info[cari][si, -2:] = possible_intersections[distance_index]
            if distance < self.car_info[cari][-1]/2:
                self.collision[cari] = True


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, car_info,sensor_info,
                 obstacle_coords,obstacle_coords1,obstacle_coords2,
                 obstacle_coords3,obstacle_coords4,):
        super(Viewer, self).__init__(width, height, resizable=False, caption='2D car', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])
        self.carn = len(car_info)
        self.car_info = []
        self.sensor_info = []
        self.car = []
        self.sensors = []
        for n in range(self.carn):
            self.car_info.append(0)
            self.sensor_info.append(0)
            self.car.append(0)
            self.sensors.append(0)
        for n in range(self.carn):
            self.car_info[n] = car_info[n]
            self.sensor_info[n] = sensor_info[n]

        self.batch = pyglet.graphics.Batch()

        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)
        for n in range(self.carn):

            self.sensors[n] = []

            line_coord = [0, 0] * 2
            color = (73, 73, 73) * 2
            for i in range(len(self.sensor_info[n])):
                self.sensors[n].append(self.batch.add(2, pyglet.gl.GL_LINES,
                                                   foreground, ('v2f', line_coord),
                                                   ('c3B', color)))

            car_box = [0, 0] * 4
            color = (249, 86, 86) * 4
            self.car[n] = self.batch.add(4, pyglet.gl.GL_QUADS,
                                      foreground, ('v2f', car_box), ('c3B', color))

        color = (134, 181, 244) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS,
                                       background, ('v2f', obstacle_coords.flatten()),
                                       ('c3B', color))
        color = (0, 0, 139) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS,
                                       background, ('v2f', obstacle_coords1.flatten()),
                                       ('c3B', color))
        color = (0, 0, 139) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS,
                                       background, ('v2f', obstacle_coords2.flatten()),
                                       ('c3B', color))
        color = (0, 0, 139) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS,
                                       background, ('v2f', obstacle_coords3.flatten()),
                                       ('c3B', color))
        color = (0, 0, 139) * 4
        self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS,
                                       background, ('v2f', obstacle_coords4.flatten()),
                                       ('c3B', color))

    def render(self):

        for n in range(self.carn):
            self._update(n)

        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    def _update(self,cari):

        cx, cy, r, w, l = self.car_info[cari]

        # sensors
        for i, sensor in enumerate(self.sensors[cari]):
            sensor.vertices = [cx, cy, *self.sensor_info[cari][i, -2:]]

        # car
        xys1 = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys1 = []
        for x, y in xys1:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys1 += [x, y]
        self.car[cari].vertices = r_xys1


if __name__ == '__main__':
    carn = 4
    np.random.seed(1)
    env = CarEnv(False, carn)
    env.set_fps(30)
    for ep in range(20):
        s = []
        r = []
        done = []
        for i in range(carn):
            s.append(env.reset(i))
            r.append(0)
            done.append(False)
        # for t in range(100):
        while True:
            for n in range(carn):
                env.render()
                if not done[n]:
                    s[n], r[n], done[n] = env.step(env.sample_action(),n)
            if not False in done:
                break


