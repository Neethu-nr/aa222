import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
from constants import *
from gym.envs.classic_control import rendering

# Reference: https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        # if self.env.hull.linearVelocity[0] ==0:
        #     self.env.done = True            
        for w in self.env.active_walkers:
            if w.face in [contact.fixtureA.body, contact.fixtureB.body]:
                # import pdb;pdb.set_trace()
                # print("face touch")
                self.env.active_walkers.remove(w)
                w.done = True
            if w.hull in [contact.fixtureA.body, contact.fixtureB.body]:
                # import pdb;pdb.set_trace()
                # print("hull touch")
                self.env.active_walkers.remove(w)
                w.done = True

            for leg in [w.legs[1], w.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = True
            for hand in [w.hands[1], w.hands[3]]:
                if hand in [contact.fixtureA.body, contact.fixtureB.body]:
                    hand.ground_contact = True

            # if w.steps%50 ==0 and w.steps>0:
            #     if w.hull.position[0] <= w.prev_loc:
            #         # import pdb;pdb.set_trace()
            #         print("no movement")
            #         self.env.active_walkers.remove(w)
            #         w.done = True
            #     else:
            #         w.prev_loc = w.hull.position[0]

    def EndContact(self, contact):
        for w in self.env.active_walkers:
            for leg in [w.legs[1], w.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = False
            for hand in [w.hands[1], w.hands[3]]:
                if hand in [contact.fixtureA.body, contact.fixtureB.body]:
                    hand.ground_contact = False

class World(gym.Env, EzPickle):
    def __init__(self):
        hardcore = False
        self.seed()
        self.viewer = None
        self.contactListener= ContactDetector(self)
        self.world = Box2D.b2World(contactListener=self.contactListener)
        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )
        self.generate_terrain(hardcore)

    def reset_world(self):
        # import pdb;pdb.set_trace()

        self.drawlist = self.terrain
        self.steps = 0
        self.scroll = max(self.pos)- VIEWPORT_W/SCALE/5
        self.done = False
        self.render()
    
    def start_population(self, genes):
        self.walkers = []
        self.reward = []
        self.pos = []
        for gene in genes:
            self.walkers.append(BipedalWalker(gene,self))
        for walker in self.walkers:
            self.reward.append(walker.total_reward)
            self.pos.append(walker.hull.position.x)
        self.active_walkers = list(self.walkers)
        self.reset_world()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.6, 0.6, 0.6)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def destroy(self):
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )


        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        for walker in self.walkers: 
            self.lidar_render = (walker.lidar_render+1) % 100
            i = walker.lidar_render
            if i < 2*len(walker.lidar):
                l = walker.lidar[i] if i < len(walker.lidar) else walker.lidar[len(walker.lidar)-i-1]
                self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for walker in self.walkers: 
            for obj in walker.drawlist:
                for f in obj.fixtures:
                    trans = f.body.transform
                    if type(f.shape) is circleShape:
                        t = rendering.Transform(translation=trans*f.shape.pos)
                        self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                    else:
                        path = [trans*v for v in f.shape.vertices]
                        self.viewer.draw_polygon(path, color=obj.color1)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array = False)

    def step(self, display):
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.steps += 1
        if self.done: return
        prev_pos = self.pos
        done = []
        self.scroll = max(self.pos)- VIEWPORT_W/SCALE/5
        self.reward = []
        self.pos = []
        self.velocity = []
        for walker in self.walkers:
            walker.step()
            self.reward.append(walker.total_reward)
            self.pos.append(walker.hull.position.x)
            self.velocity.append(walker.hull.linearVelocity[0])
            done.append(walker.done)
        if display:
            self.render()

        # stopping condition
        if all(done) or (self.pos == prev_pos and any(self.velocity)):
            self.done = True

class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self, gene, World):
        EzPickle.__init__(self)
        self.seed()
        self.world = World.world
        self.steps = 0
        self.prev_loc = None
        self.MOTORS_TORQUE, self.SPEED_HIP, self.SPEED_KNEE, self.HULL_H, self.HULL_W = gene

        self.FACE_H = self.HULL_H*FACE_HULL_SCALE
        self.FACE_W =  self.HULL_W*FACE_HULL_SCALE

        # self.terrain = terrain
        self.hull = None
        self.face = None

        self.prev_shaping = None

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1,-1, -1, -1, -1]), np.array([1, 1, 1, 1,1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.total_reward = 0
        self.a = np.array([0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0])
        self.state = STAY_ON_ONE_LEG
        self.moving_leg = 0
        self.supporting_leg = 1 - self.moving_leg
        self.supporting_knee_angle = SUPPORT_KNEE_ANGLE
        self.done = False
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        # if self.terrain is None: return
        # self.terrain = None
        if not self.hull: return
        self.world.contactListener = None
        self.world.DestroyBody(self.hull)
        self.world.DestroyBody(self.face)
        self.hull = None
        self.face = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        for hand in self.hands:
            self.world.DestroyBody(hand)
        self.hands = []
        self.joints = []    # joints are destroyed with attached bodies 

    def reset(self):
        self._destroy()
        self.prev_shaping = None
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE


        self.joints = []
        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+2*LEG_H
        HULL_POLY =[
            (-self.HULL_W/2,-self.HULL_H/2), ( self.HULL_W/2,-self.HULL_H/2),
            (-self.HULL_W/2, self.HULL_H/2), ( self.HULL_W/2, self.HULL_H/2)
            ]
        FACE_POLY =[
            (-self.FACE_W/2,-self.FACE_H/2), ( self.FACE_W/2,-self.FACE_H/2),
            (-self.FACE_W/2, self.FACE_H/2), ( self.FACE_W/2, self.FACE_H/2)
            ]

        FACE_FD = fixtureDef(
                        shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in FACE_POLY ]),
                        density=5.0,
                        friction=0.1,
                        categoryBits=0x0020,
                        maskBits=0x001,  # collide only with ground
                        restitution=0.0) # 0.99 bouncy

        HULL_FD = fixtureDef(
                        shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                        density=5.0,
                        friction=0.1,
                        categoryBits=0x0020,
                        maskBits=0x001,  # collide only with ground
                        restitution=0.0) # 0.99 bouncy

        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)
        self.face = None
        # self.face = self.world.CreateDynamicBody(
        #     position = (init_x, init_y+self.HULL_H/SCALE+self.FACE_H/SCALE),
        #     fixtures = FACE_FD
        #         )
        # self.face.color1 = (0.4,0.2,0.9)
        # self.face.color2 = (0.4,0.2,0.5)
        # self.face.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)
        # rjd = revoluteJointDef(
        #     bodyA=self.face,
        #     bodyB=self.hull,
        #     localAnchorA=(0, -self.FACE_H/2/SCALE),
        #     localAnchorB=(0, self.HULL_H/2/SCALE),
        #     enableMotor=True,
        #     maxMotorTorque=self.MOTORS_TORQUE,
        #     motorSpeed = 1,
        #     enableLimit = True,
        #     lowerAngle = -0.25,
        #     upperAngle = 0.5,
        #     )
        # self.joints.append(self.world.CreateJoint(rjd))

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = LEG_FD
                )
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
                )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = LOWER_FD
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))


        self.hands = []
        for i in [+1,-1]:
            hand = self.world.CreateDynamicBody(
                position = (init_x + i*self.HULL_W/2/SCALE, init_y + self.HULL_H/SCALE*0.75 ),
                angle = (i*0.05),
                fixtures = HAND_FD
                )
            hand.color1 = (0.6-i/10., 0.1-i/10., 0.5-i/10.)
            hand.color2 = (0.2-i/10., 0.1-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=hand,
                localAnchorA=( i*self.HULL_W/2/SCALE*0.5, self.HULL_H/SCALE*0.75 + LEG_DOWN),
                localAnchorB=(0, HAND_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -1.5,
                upperAngle = 1.5,
                )
            self.hands.append(hand)
            self.joints.append(self.world.CreateJoint(rjd))
 
            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y + self.HULL_H/SCALE*0.75 -HAND_H*3/2 ),
                angle = (i*0.05),
                fixtures = HAND_LOWER_FD
                )
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=hand,
                bodyB=lower,
                localAnchorA=(0, -HAND_H/2),
                localAnchorB=(0, HAND_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = 1.5,
                )
            lower.ground_contact = False
            self.hands.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))
        self.drawlist =  self.legs +self.hands[:2]+ [self.hull, self.face]+ self.hands[2:]
        self.drawlist =  self.legs +self.hands[:2]+ [self.hull]+ self.hands[2:]
        self.prev_loc = self.hull.position[0]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction
        self.lidar = [LidarCallback() for _ in range(10)]

        return self._step(self.a)[0]

    def _step(self, action):
        if self.done:
            return

        self.hull.ApplyForceToCenter((0, 20), True) #-- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(self.SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(self.SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(self.SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(self.SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(self.SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(self.SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(self.SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(self.SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

            self.joints[6].motorSpeed     = float(self.SPEED_HIP     * np.sign(action[4]))
            self.joints[6].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[4]), 0, 1))
            self.joints[7].motorSpeed     = float(self.SPEED_KNEE    * np.sign(action[5]))
            self.joints[7].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[5]), 0, 1))
            self.joints[4].motorSpeed     = float(self.SPEED_HIP     * np.sign(action[6]))
            self.joints[4].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[6]), 0, 1))
            self.joints[5].motorSpeed     = float(self.SPEED_KNEE    * np.sign(action[7]))
            self.joints[5].maxMotorTorque = float(self.MOTORS_TORQUE * np.clip(np.abs(action[7]), 0, 1))


        # self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / self.SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / self.SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / self.SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / self.SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==24

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * self.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.done or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def step(self):
        if self.done:
            self.drawlist = [] 
            return

        s, r, self.done = self._step(self.a)
        self.steps += 1
        self.total_reward += r

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*self.moving_leg
        supporting_s_base = 4 + 5*self.supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if self.state==STAY_ON_ONE_LEG:
            hip_targ[self.moving_leg]  = 1.1
            knee_targ[self.moving_leg] = -0.6
            self.supporting_knee_angle += 0.03
            if s[2] > SPEED: self.supporting_knee_angle += 0.03
            self.supporting_knee_angle = min( self.supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                self.state = PUT_OTHER_DOWN
        if self.state==PUT_OTHER_DOWN:
            hip_targ[self.moving_leg]  = +0.1
            knee_targ[self.moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[self.supporting_leg] = self.supporting_knee_angle
            if s[moving_s_base+4]:
                self.state = PUSH_OFF
                self.supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if self.state==PUSH_OFF:
            knee_targ[self.moving_leg] = self.supporting_knee_angle
            knee_targ[self.supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                self.state = STAY_ON_ONE_LEG
                self.moving_leg = 1 - self.moving_leg
                self.supporting_leg = 1 - self.moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        self.a[0] = hip_todo[0]
        self.a[1] = knee_todo[0]
        self.a[2] = hip_todo[1]
        self.a[3] = knee_todo[1]

        self.a[4] = hip_todo[0]
        self.a[5] = knee_todo[0]
        self.a[6] = hip_todo[1]
        self.a[7] = knee_todo[1]
        self.a = np.clip(0.5*self.a, -1.0, 1.0)

def test_walkers(genes):
    print(genes)
    # genes = [[84, 2, 2, 42, 31], [55, 6, 5, 44, 37], [33, 6, 5, 6, 33], [35, 3, 3, 43, 43], 
    # [13, 1, 1, 40, 5], [27, 4, 6, 31, 41], [54, 9, 9, 23, 35], [47, 7, 3, 2, 29], [69, 5, 2, 41, 12], [34, 2, 5, 22, 32]]
    # genes = [[80,4,6,30,28], [80,4,6,30,48],[80,4,6,50,28]] 
    # Heurisic: suboptimal, have no notion of balance.
    # import pdb;pdb.set_trace()
    world = World(genes)
    while not world.done:         
        # import pdb;pdb.set_trace()
        world.step()
    print("all walkers have stopped, final rewards:", world.reward)
    return world.reward
class Fitness:
    def __init__(self):
        self.world = World()

    def fitness(self, genes, display=False):
        self.steps = 0
        self.world.start_population(genes)
        while not self.world.done and self.steps<500:          
            self.steps += 1
            self.world.step(display)
        # print("all walkers have stopped, final rewards:", self.world.reward)
        return self.world.reward
    
class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True
if __name__=="__main__":
    genes = [[84,4,6,30,26.5]]
    test_walkers(genes)
