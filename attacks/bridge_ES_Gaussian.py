#!/usr/bin/env python3
import argparse
import math
import os
import re
import signal
import threading
import time
from multiprocessing import Process, Queue, Manager
from typing import Any
import random
import cv2
from PIL import Image
from threading import Thread

import carla  # pylint: disable=import-error
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc import VisionIpcServer, VisionStreamType
from common.basedir import BASEDIR
from common.numpy_fast import clip
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
from tools.sim.lib.can import can_function

import gc
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
import pickle
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

W, H = 1928, 1208
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'accelerometer', 'gyroscope', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState', "modelV2"])

# Generar el parche inicial de 100x100 píxeles con valores RGB aleatorios
SAVE_FRAMES = None # None -> Not save frames ; int --> save frame each <int> frames f.e. 100
ADVERSARIAL_DIR = "/home/ikerlan/Unai/openpilot/selfdrive/modeld/adversarial"
AUTO_PILOT = True
NUM_ACTORS = 5
GENERATE_PATCH = True # True -> patch training ; False -> Normal simulation
SAVE_PATCH =  "patch" # None -> Not save patch ; str -> filename without extension (will be saved in .png and .npy)
LEAD_CONF = -1
LEAD_DIST = -1
DATA_LIST = []
CURRENT_FRAME = 0
STOP_FRAME = 8000
PATCH_SIZE = (50,50)
LEARNING_RATE = 50
STRENGTH = 25
MAX_PATCH = 0
OPENPILOT_VEHICLE = None
ADVERSARIAL_VEHICLE = None
TRAFFIC_APPLIED = False
REAL_DATA = []
REAL_SIM = False
ACT_INDEX = 0

if os.path.exists(ADVERSARIAL_DIR+"/patches/npy"):
  numbers = []
  for filename in os.listdir(ADVERSARIAL_DIR+"/patches/npy"):
    if filename.endswith('.pkl'):
      match = re.search(r'\d+', filename)
      if match:
          number = int(match.group())
          numbers.append(number)
  if len(numbers) > 0:
    MAX_PATCH = max(numbers)

def load_patch(filename=None):
  global MAX_PATCH
  MAX_PATCH = 0
  if filename is None:
    numbers = []
    for filename in os.listdir(ADVERSARIAL_DIR+"/patches/npy"):
      if filename.endswith('.pkl'):
        match = re.search(r'\d+', filename)
        if match:
            number = int(match.group())
            numbers.append(number)
    if len(numbers) > 0:
      MAX_PATCH = max(numbers)
    filename = ADVERSARIAL_DIR+"/patches/npy/patch_"+str(MAX_PATCH)+".pkl"
  else:
    filename = ADVERSARIAL_DIR+"/patches/npy/"+filename
  if os.path.exists(filename):
    with open(filename, 'rb') as file:
      patch = pickle.load(file)
  else:
    patch = [np.random.randint(0, 256, (PATCH_SIZE[0], PATCH_SIZE[1], 3), dtype=np.uint8), 999999, 999999, 999999]
  return patch

def initialize_dirs():
  if not os.path.exists(ADVERSARIAL_DIR+'/patches'):
    os.makedirs(ADVERSARIAL_DIR+"/patches")
  if not os.path.exists(ADVERSARIAL_DIR+'/patches/npy'):
    os.makedirs(ADVERSARIAL_DIR+"/patches/npy")
  if not os.path.exists(ADVERSARIAL_DIR+'/patches/png'):
    os.makedirs(ADVERSARIAL_DIR+"/patches/png")
  if not os.path.exists(ADVERSARIAL_DIR+'/frames/'):
    os.makedirs(ADVERSARIAL_DIR+'/frames/')

# Initialize used directories
initialize_dirs()
# Cargar datos de la simulacion sin parche
if os.path.exists(ADVERSARIAL_DIR+"/patches/npy/real_sim.pkl"):
  with open(ADVERSARIAL_DIR+"/patches/npy/real_sim.pkl", 'rb') as file:
    REAL_DATA = pickle.load(file)[1]
else:
  REAL_SIM = True
# Load actual simulation patch in use
if os.path.exists(ADVERSARIAL_DIR+"/patches/npy/patch_act.pkl"):
  with open(ADVERSARIAL_DIR+"/patches/npy/patch_act.pkl", 'rb') as file:
      patch_act = pickle.load(file)
else:
  patch_act = np.random.randint(0, 256, (PATCH_SIZE[0], PATCH_SIZE[1], 3), dtype=np.uint8)
# Load previous patch
if os.path.exists(ADVERSARIAL_DIR+"/patches/npy/patch_"+str(MAX_PATCH)+".pkl"):
  with open(ADVERSARIAL_DIR+"/patches/npy/patch_"+str(MAX_PATCH)+".pkl", 'rb') as file:
    patch_prev_info = pickle.load(file)
    patch_prev = patch_prev_info[0]

def random_update(patch, num_pixels, strength=25):
  for _ in range(num_pixels):
    x = np.random.randint(0, PATCH_SIZE[0])
    y = np.random.randint(0, PATCH_SIZE[1])
    x_min = max(x - 1, 0)
    x_max = min(x + 2, patch.shape[0])
    y_min = max(y - 1, 0)
    y_max = min(y + 2, patch.shape[1])
    adjacent_pixels = patch[x_min:x_max, y_min:y_max]
    mean_pixel = np.mean(adjacent_pixels, axis=(0, 1)).astype(np.uint8)
    # cojer un random entre [mean-25, mean+25]
    pixel_min = np.clip(mean_pixel - strength, 0, 255)
    pixel_max = np.clip(mean_pixel + strength, 0, 256)  # 256 to include 255 in randint
    new_pixel = np.random.randint(pixel_min, pixel_max, dtype=np.uint8)
    patch[x, y] = new_pixel
  return patch

def disappearance_loss(image, conf, dist, real_dist, l1=0.01, l2=0.001):
    """
    Calculate the disappearance loss using confidence scores, detected distances, and total variation.

    Parameters:
    - image: The adversarial patch image of shape (x, y, 3).
    - conf: Confidence score from the model output with the adversarial patch (float).
    - dist: Detected distance from the model output with the adversarial patch.
    - real_dist: Detected distance from the model output without the adversarial patch.
    - l1: Weight for the distance term.
    - l2: Weight for the Total Variation loss.

    Returns:
    - Loss value.
    """
    # Confidence loss component
    lconf = -math.log(1-conf)
    # Pérdida de distancia: fomentar una mayor distancia percibida para el líder
    #ldis = torch.mean(torch.abs(dist - prev_dist) / prev_dist)
    ldis = -abs(dist / real_dist)
    # Pérdida de variación total (TV): se calcula sobre el parche, se asumirá que se calcula en otro lugar
    #ltv = torch.sum(torch.abs(image[:, :, 1:] - image[:, :, :-1])) + torch.sum(torch.abs(image[:, 1:, :] - image[:, :-1, :]))
    ind = np.arange(0, 50-1)
    ltv = np.sum(abs(image[ind+1, :, :] - image[ind, :, :])) + np.sum(abs(image[:, ind+1, :] - image[:, ind, :]))
    # Combinar las pérdidas con los pesos dados
    loss = lconf + l1 * ldis + l2 * ltv
    return loss

def load_distances():
  loaded_data = []
  try:
    filepath = ADVERSARIAL_DIR + "/patches/distances.pkl"
    with open(filepath, 'rb') as file:
      loaded_data = pickle.load(file)
  except Exception as e:
    print(e)
  return loaded_data

def save_distances(distances):
  filepath = ADVERSARIAL_DIR + "/patches/distances.pkl"
  with open(filepath, 'wb') as file:
    pickle.dump(distances, file)

def gaussian_mutation(image, sigma=0.05, blur_sigma=5):
    """
    Apply Gaussian mutation to an image.
    """
    noise = np.random.normal(0, sigma, image.shape)
    smooth_noise = gaussian_filter(noise, sigma=blur_sigma)
    mutated_image = image + smooth_noise
    mutated_image = np.clip(mutated_image, 0, 255).astype(np.uint8)
    return mutated_image

def one_plus_one_evolution_strategy_algorithm(data_list, lr=100, sgth=25):
  # INPUT: [(prev_dist, lead_dist, conf, loss)]
  # media de cada distances_list
  d_mean = np.mean([elem[2] for elem in data_list]) 
  c_mean = np.mean([elem[3] for elem in data_list])
  l_mean = np.mean([elem[4] for elem in data_list])
  d_mean_prev = np.mean([elem[2] for elem in patch_prev_info[1]])
  c_mean_prev = np.mean([elem[3] for elem in patch_prev_info[1]])
  l_mean_prev = np.mean([elem[4] for elem in patch_prev_info[1]])
  print(f"{d_mean} ; {c_mean} ; {l_mean}") # medidas de patch_act
  print(f"{d_mean_prev} ; {c_mean_prev} ; {l_mean_prev}") # medidas de patch_prev
  # comparar con patch_prev_info[1:]
  if ((c_mean < c_mean_prev)) or patch_prev is None:
    patch_next = gaussian_mutation(patch_act, lr)
    with open(ADVERSARIAL_DIR+'/patches/npy/'+str(SAVE_PATCH)+'_'+str(MAX_PATCH+1)+'.pkl', 'wb') as file:
      pickle.dump([patch_act, data_list], file)
    print("[*] Patch successfully updated!")
  else:
    patch_next = gaussian_mutation(patch_prev, lr)
    print("[!] Patch not updated!")
  with open(ADVERSARIAL_DIR+'/patches/npy/patch_act.pkl', 'wb') as file:
    pickle.dump(patch_next, file)

def find_nearest_coordinate(coords, target, index=0):
    """
    Finds the nearest coordinate to the target in a list of coordinates.
    
    Args:
    coords (list of tuples): List of coordinates (x, y, z).
    target (tuple): The target coordinate (x, y, z).
    
    Returns:
    tuple: The nearest coordinate to the target.
    """
    min_distance = float('inf')  # Initialize with a very large number
    nearest_coord = None
    best_index = None
    
    # Calculate distance from target to each coordinate in the list
    for i in range(index, len(coords)):
        distance = math.sqrt((coords[i][0][0] - target[0])**2 + (coords[i][0][1] - target[1])**2 + (coords[i][0][2] - target[2])**2)
        if distance < 2:
          return i, coords[i]
        if distance < min_distance:
            min_distance = distance
            nearest_coord = coords[i]
            best_index = i
    return best_index, nearest_coord

# save the camera frames as PNG
def save_image(image_data, file_name):
  """Function to save an image file."""
  try:
    if image_data.dtype != np.uint8:
      image_data = (image_data * 255).astype(np.uint8) # fixed: Error, could not save image:  Cannot handle this data type: (1, 1, 3), <f4
      # Solution: Specifically, it appears that the array's dtype is float32 (indicated by <f4>), which is not directly supported by the PIL library for image saving. Converting the array to a supported dtype, such as uint8, should resolve the issue
    img_save = Image.fromarray(image_data)
    img_save.save(file_name)
    print(f"Image successfully saved: {file_name}")
  except Exception as e:
    print("Error, could not save image: ", e)

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--dual_camera', action='store_true')
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=16)
  parser.add_argument('--host', dest='host', type=str, default='127.0.0.1')
  parser.add_argument('--port', dest='port', type=int, default=2000)

  return parser.parse_args(add_args)

# https://carla.readthedocs.io/en/latest/tuto_G_traffic_manager/
# https://carla.readthedocs.io/en/docs-preview/adv_traffic_manager/
# https://carla.readthedocs.io/en/latest/python_api/#carlatrafficmanager
def applyTrafficManagerSettings(world, client, auto_drive=True):
  traffic_manager = client.get_trafficmanager()
  traffic_manager.set_synchronous_mode(True)
  ego_vehicle = getEgoVehicle(world, client)
  for vehicle in world.get_actors().filter('*vehicle*'):
    if vehicle.id != ego_vehicle.id:
      vehicle.set_autopilot(auto_drive)
    """traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    #traffic_manager.set_random_device_seed(0)
    for vehicle in world.get_actors().filter('*vehicle*'):
      ego_vehicle = getEgoVehicle(world, client)
      if vehicle.id != ego_vehicle.id:
        vehicle.set_autopilot(auto_drive)
      traffic_manager.ignore_lights_percentage(vehicle, random.randint(0,10))
      traffic_manager.distance_to_leading_vehicle(vehicle, 3)
      traffic_manager.vehicle_percentage_speed_difference(vehicle, 20)
      traffic_manager.auto_lane_change(vehicle, True)
      #traffic_manager.update_vehicle_lights(vehicle, True)
      traffic_manager.ignore_walkers_percentage(vehicle, 0)"""

def spawnRandomActors(world, actor, numActors):
    bpl = world.get_blueprint_library()
    actor_blueprints = bpl.filter(actor)
    spawn_points = world.get_map().get_spawn_points() # spawn_points es un array de carla.Transform()
    #for i, spawn_point in enumerate(spawn_points):
    #    world.debug.draw_string(spawn_point.location, str(i), life_time=10)
    maxActors = min(len(spawn_points), numActors)
    print("Number of actors to spawn: " + str(maxActors))
    spawnedActors = []
    for i, spawn_point in enumerate(random.sample(spawn_points, maxActors)):
        try:
            actor = world.try_spawn_actor(random.choice(actor_blueprints), spawn_point) # Equivalente a command.SpawnActor()
            world.tick()
            if actor:
                spawnedActors.append(actor)
        except Exception as e:
            print(e)
    return spawnedActors

def getEgoVehicle(world, client):
    for actor in world.get_actors().filter('vehicle.*'):
        if hasattr(actor, 'attributes') and 'role_name' in actor.attributes:
            role_name = actor.attributes['role_name']
            if role_name == 'hero':
                print(f"Found hero vehicle (ID {actor.id})")
                return actor
    return None

class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new


class Camerad:
  def __init__(self, dual_camera):
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    if dual_camera:
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    kernel_fn = os.path.join(BASEDIR, "tools/sim/rgb_to_nv12.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_nv12
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback_road(self, image):
    self._cam_callback(image, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    self._cam_callback(image, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def _cam_callback(self, image, frame_id, pub_type, yuv_type):
    #
    # INPUT
    #print(type(image)) --> <class 'carla.libcarla.Image'>
    #print(image) --> Image(frame=1128, timestamp=5.395764, size=1928x1208)
    #print(image.raw_data)
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (H, W, 4))
    img = img[:, :, [0, 1, 2]].copy()

    #if frame_id % 25 == 0:
    if SAVE_FRAMES and frame_id % SAVE_FRAMES == 0:
      thread = threading.Thread(target=save_image, args=(img, ADVERSARIAL_DIR+'/frames/'+str(image.frame)+".png"))
      thread.start()
      THREADS.append(thread)
      print("saving image..."+str(image.frame))

    # convert RGB frame to YUV
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    eof = int(frame_id * 0.05 * 1e9)

    self.vipc_server.send(yuv_type, yuv.data.tobytes(), frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    pm.send(pub_type, dat)

def imu_callback(imu, vehicle_state):
  # send 5x since 'sensor_tick' doesn't seem to work. limited by the world tick?
  for _ in range(5):
    vehicle_state.bearing_deg = math.degrees(imu.compass)
    dat = messaging.new_message('accelerometer')
    dat.accelerometer.sensor = 4
    dat.accelerometer.type = 0x10
    dat.accelerometer.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
    pm.send('accelerometer', dat)

    # copied these numbers from locationd
    dat = messaging.new_message('gyroscope')
    dat.gyroscope.sensor = 5
    dat.gyroscope.type = 0x10
    dat.gyroscope.timestamp = dat.logMonoTime  # TODO: use the IMU timestamp
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
    pm.send('gyroscope', dat)
    time.sleep(0.01)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "unixTimestampMillis": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverStateV2', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverStateV2')
    dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.leftDriverData.faceProb = 1.0
    dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
    dat.driverStateV2.rightDriverData.faceProb = 1.0
    pm.send('driverStateV2', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  i = 0
  while not exit_event.is_set():
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1


def connect_carla_client(host: str, port: int):
  client = carla.Client(host, port)
  client.set_timeout(5)
  return client


class CarlaBridge:

  def __init__(self, arguments):
    set_params_enabled()

    self.params = Params()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.params.put("CalibrationParams", msg.to_bytes())
    self.params.put_bool("DisengageOnAccelerator", True)

    self._args = arguments
    self._carla_objects = []
    self._camerad = None
    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()

  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      while self._keep_alive:
        try:
          self._run(q)
          break
        except RuntimeError as e:
          self.close()
          if retries == 0:
            raise

          # Reset for another try
          self._carla_objects = []
          self._threads = []
          self._exit_event = threading.Event()

          retries -= 1
          if retries <= -1:
            print(f"Restarting bridge. Error: {e} ")
          else:
            print(f"Restarting bridge. Retries left {retries}. Error: {e} ")
    finally:
      # Clean up resources in the opposite order they were created.
      self.close()

  def _run(self, q: Queue):
    client = connect_carla_client(self._args.host, self._args.port)
    world = client.load_world(self._args.town)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    if not self._args.high_quality:
      world.unload_map_layer(carla.MapLayer.Foliage)
      world.unload_map_layer(carla.MapLayer.Buildings)
      world.unload_map_layer(carla.MapLayer.ParkedVehicles)
      world.unload_map_layer(carla.MapLayer.Props)
      world.unload_map_layer(carla.MapLayer.StreetLights)
      world.unload_map_layer(carla.MapLayer.Particles)

    blueprint_library = world.get_blueprint_library()

    world_map = world.get_map()

    #############################
    # Spawn adversarial vehicle #
    #############################
    bp = blueprint_library.find('vehicle.carlamotors.carlacola')
    for idx,spawn in enumerate(world.get_map().get_spawn_points()):
      print(f"{idx}: {spawn}")
    transform = world.get_map().get_spawn_points()[0]
    global ADVERSARIAL_VEHICLE
    ADVERSARIAL_VEHICLE = world.spawn_actor(bp, transform)
    self._carla_objects.append(ADVERSARIAL_VEHICLE)
    print(f"CARLACOLA: {transform}")

    ###########################
    # Spawn openpilot vehicle #
    ###########################
    vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
    vehicle_bp.set_attribute('role_name', 'hero')
    spawn_points = world_map.get_spawn_points()
    assert len(spawn_points) > self._args.num_selected_spawn_point, f'''No spawn point {self._args.num_selected_spawn_point}, try a value between 0 and
      {len(spawn_points)} for this town.'''
    spawn_point = spawn_points[self._args.num_selected_spawn_point]
    print(f"OPENPILOT: {spawn_point}")
    global OPENPILOT_VEHICLE
    OPENPILOT_VEHICLE = world.spawn_actor(vehicle_bp, spawn_point)
    self._carla_objects.append(OPENPILOT_VEHICLE)
    max_steer_angle = OPENPILOT_VEHICLE.get_physics_control().wheels[0].max_steer_angle

    spawnedActors = spawnRandomActors(world, '*vehicle*', NUM_ACTORS)
    print(AUTO_PILOT)
    start_time = time.time()
    
    # Camera
    """camera_bp = blueprint_library.find('sensor.camera.depth')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    self._carla_objects.append(camera)"""

    # make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = OPENPILOT_VEHICLE.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    OPENPILOT_VEHICLE.apply_physics_control(physics_control)

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      if not self._args.high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=OPENPILOT_VEHICLE)
      camera.listen(callback)
      return camera

    self._camerad = Camerad(self._args.dual_camera)

    if self._args.dual_camera:
      road_wide_camera = create_camera(fov=120, callback=self._camerad.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts
      self._carla_objects.append(road_wide_camera)
    road_camera = create_camera(fov=40, callback=self._camerad.cam_callback_road)
    self._carla_objects.append(road_camera)

    vehicle_state = VehicleState()

    # re-enable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.01')
    imu = world.spawn_actor(imu_bp, transform, attach_to=OPENPILOT_VEHICLE)
    imu.listen(lambda imu: imu_callback(imu, vehicle_state))

    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, transform, attach_to=OPENPILOT_VEHICLE)
    gps.listen(lambda gps: gps_callback(gps, vehicle_state))
    self.params.put_bool("UbloxAvailable", True)

    self._carla_objects.extend([imu, gps])
    # launch fake car threads
    self._threads.append(threading.Thread(target=panda_state_function, args=(vehicle_state, self._exit_event,)))
    self._threads.append(threading.Thread(target=peripheral_state_function, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=fake_driver_monitoring, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, self._exit_event,)))
    for t in self._threads:
      t.start()

    # init
    throttle_ease_out_counter = REPEAT_COUNTER
    brake_ease_out_counter = REPEAT_COUNTER
    steer_ease_out_counter = REPEAT_COUNTER

    vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

    is_openpilot_engaged = False
    throttle_out = steer_out = brake_out = 0.
    throttle_op = steer_op = brake_op = 0.
    throttle_manual = steer_manual = brake_manual = 0.

    old_steer = old_brake = old_throttle = 0.
    throttle_manual_multiplier = 0.7  # keyboard signal is always 1
    brake_manual_multiplier = 0.7  # keyboard signal is always 1
    steer_manual_multiplier = 45 * STEER_RATIO  # keyboard signal is always 1

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      world.tick()

    # loop
    rk = Ratekeeper(100, print_delay_threshold=0.05)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    while self._keep_alive:

      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can

      cruise_button = 0
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0
      throttle_manual = steer_manual = brake_manual = 0.0

      # --------------Step 1-------------------------------
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          steer_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "throttle":
          throttle_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "brake":
          brake_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "reverse":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        elif m[0] == "cruise":
          if m[1] == "down":
            cruise_button = CruiseButtons.DECEL_SET
            is_openpilot_engaged = True
          elif m[1] == "up":
            cruise_button = CruiseButtons.RES_ACCEL
            is_openpilot_engaged = True
          elif m[1] == "cancel":
            cruise_button = CruiseButtons.CANCEL
            is_openpilot_engaged = False
        elif m[0] == "ignition":
          vehicle_state.ignition = not vehicle_state.ignition
        elif m[0] == "quit":
          break

        throttle_out = throttle_manual * throttle_manual_multiplier
        steer_out = steer_manual * steer_manual_multiplier
        brake_out = brake_manual * brake_manual_multiplier

        old_steer = steer_out
        old_throttle = throttle_out
        old_brake = brake_out
      global CURRENT_FRAME
      if is_openpilot_engaged:
        # Start moving Adversarial Vehicle only if Openpilot is engaged
        global TRAFFIC_APPLIED
        if not TRAFFIC_APPLIED:
          # Coprobar que openpilot se ha empezado a mover para que ambos vehiculos empiecen a moverse a la vez (igualdad de situaciones en todas las simulaciones)
          if OPENPILOT_VEHICLE.get_velocity().x > 1 or OPENPILOT_VEHICLE.get_velocity().y > 1 or OPENPILOT_VEHICLE.get_velocity().z > 1:
            TRAFFIC_APPLIED = True
            applyTrafficManagerSettings(world, client, AUTO_PILOT)

        sm.update(0)

        # TODO gas and brake is deprecated
        throttle_op = clip(sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = sm['carControl'].actuators.steeringAngleDeg
        if GENERATE_PATCH:
          leads = sm['modelV2'].leadsV3
          global patch_act
          global LEAD_CONF
          global LEAD_DIST
          global DATA_LIST
          global ACT_INDEX
          if len(leads) > 0:
            # TODO: realmente la buena es leads[0].prob???, igual habria que coger la maxima de todas
            #print(f"{leads[0].prob} ; {leads[0].x[0]}")
            LEAD_CONF = leads[0].prob
            #if leads[0].x[0] != LEAD_DIST:
            LEAD_DIST = leads[0].x[0]
            op_loc = OPENPILOT_VEHICLE.get_location()
            adv_loc = ADVERSARIAL_VEHICLE.get_location()
            op_loc_coords = (op_loc.x, op_loc.y, op_loc.z)
            adv_loc_coords = (adv_loc.x, adv_loc.y, adv_loc.z)
            if not REAL_SIM and LEAD_DIST < 40: # Si simulación con parche
              # TODO: Comprobar location de ambos vehiculos en el real y en el actual, si estan en el mismo location, entonces comparar distancias detectadas
              real_dist = None
              ACT_INDEX, nearest_real_coord = find_nearest_coordinate(REAL_DATA,op_loc_coords,ACT_INDEX)
              #if abs(sum(abs(a - b) for a, b in zip(op_loc_coords, nearest_real_coord[0]))) < 10:
              #  real_dist = nearest_real_coord[2]
              real_dist = nearest_real_coord[2]
              #if real_dist is None:
              #  print("[!] No se ha podido encontrar la distancia real")
              #  while True:
              #    None
              loss = disappearance_loss(patch_act, LEAD_CONF, LEAD_DIST, real_dist)
            else: # Primera simulacion sin parche
              loss = 999999
              real_dist = LEAD_DIST
            if real_dist < 40: # ???? si el parche surge efecto, ese 50 se vera afectado respecto a la simulacion real para las localizaciones
              DATA_LIST.append((op_loc_coords, adv_loc_coords, LEAD_DIST, LEAD_CONF, loss))
            #print(DATA_LIST)
          else:
            LEAD_CONF = -1
            LEAD_DIST = -1
        #print("#####################################################################################################################################################")

        throttle_out = throttle_op
        steer_out = steer_op
        brake_out = brake_op

        steer_out = steer_rate_limit(old_steer, steer_out)
        old_steer = steer_out

      else:
        if throttle_out == 0 and old_throttle > 0:
          if throttle_ease_out_counter > 0:
            throttle_out = old_throttle
            throttle_ease_out_counter += -1
          else:
            throttle_ease_out_counter = REPEAT_COUNTER
            old_throttle = 0

        if brake_out == 0 and old_brake > 0:
          if brake_ease_out_counter > 0:
            brake_out = old_brake
            brake_ease_out_counter += -1
          else:
            brake_ease_out_counter = REPEAT_COUNTER
            old_brake = 0

        if steer_out == 0 and old_steer != 0:
          if steer_ease_out_counter > 0:
            steer_out = old_steer
            steer_ease_out_counter += -1
          else:
            steer_ease_out_counter = REPEAT_COUNTER
            old_steer = 0

      # --------------Step 2-------------------------------
      steer_carla = steer_out / (max_steer_angle * STEER_RATIO * -1)

      steer_carla = np.clip(steer_carla, -1, 1)
      steer_out = steer_carla * (max_steer_angle * STEER_RATIO * -1)
      old_steer = steer_carla * (max_steer_angle * STEER_RATIO * -1)

      vc.throttle = throttle_out / 0.6
      vc.steer = steer_carla
      vc.brake = brake_out
      OPENPILOT_VEHICLE.apply_control(vc)

      # --------------Step 3-------------------------------
      vel = OPENPILOT_VEHICLE.get_velocity()
      speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # in m/s
      vehicle_state.speed = speed
      vehicle_state.vel = vel
      vehicle_state.angle = steer_out
      vehicle_state.cruise_button = cruise_button
      vehicle_state.is_engaged = is_openpilot_engaged

      CURRENT_FRAME = rk.frame
      if CURRENT_FRAME % 100 == 0:
        print("[*] Frame: "+str(CURRENT_FRAME))
      if CURRENT_FRAME == STOP_FRAME:
        if not REAL_SIM:
          one_plus_one_evolution_strategy_algorithm(DATA_LIST, LEARNING_RATE, STRENGTH)
        else:
          with open(ADVERSARIAL_DIR+'/patches/npy/'+str(SAVE_PATCH)+'_'+str(MAX_PATCH)+'.pkl', 'wb') as file:
            pickle.dump([None, DATA_LIST], file)
          with open(ADVERSARIAL_DIR+'/patches/npy/real_sim.pkl', 'wb') as file:
            pickle.dump([None, DATA_LIST], file)
          with open(ADVERSARIAL_DIR+'/patches/npy/patch_act.pkl', 'wb') as file:
            pickle.dump(patch_act, file)
        print("#################### PATCH SAVED ####################")
        sys.exit(0)
        while True:
          None
      if rk.frame % PRINT_DECIMATION == 0:
        print("frame: ", rk.frame, "; speed (m/s)", speed, "; engaged:", is_openpilot_engaged, "; throttle: ", round(vc.throttle, 3), "; steer(c/deg): ",
              round(vc.steer, 3), round(steer_out, 3), "; brake: ", round(vc.brake, 3))
      if rk.frame % 5 == 0:
        world.tick()
      rk.keep_time()
      self.started = True

  def close(self):
    self.started = False
    self._exit_event.set()

    for s in self._carla_objects:
      try:
        s.destroy()
      except Exception as e:
        print("Failed to destroy carla object", e)
    for t in reversed(self._threads):
      t.join()

  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries))
    bridge_p.start()
    return bridge_p


if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()

  carla_bridge = CarlaBridge(args)
  p = carla_bridge.run(q)

  if args.joystick:
    # start input poll for joystick
    from tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(q)
  for thread in THREADS:
    thread.join()
  p.join()
