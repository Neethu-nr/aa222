from Box2D.b2 import (fixtureDef, polygonShape)

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
LIDAR_RANGE   = 160/SCALE
INITIAL_RANDOM = 5
FACE_HULL_SCALE = 0.5

LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE
HAND_W, HAND_H = LEG_W*0.75, LEG_H*0.75

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5
STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
SPEED = 0.29  # Will fall forward on higher speed
SUPPORT_KNEE_ANGLE = +0.1


LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

HAND_FD = fixtureDef(
                    shape=polygonShape(box=(HAND_W/2, HAND_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

HAND_LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*HAND_W/2, HAND_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
