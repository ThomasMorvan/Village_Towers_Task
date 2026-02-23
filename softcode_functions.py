import time
from PyQt5.QtGui import QColor
from sound_functions import sound_device, tone_generator, whitenoise_generator
from video_functions import (
    draw_square_generator,
)
from village.manager import manager


# genero un sonido y lo cargo en el dispositivo de sonido con el mismo volumen en
# ambos canales
def function1():
    gain = 0.05
    sound = tone_generator(duration=1, frequency=2000, ramp_time=0.005)
    sound *= gain
    sound_device.load(sound, sound)

# generamos un sonido de ruido blanco y hacemos que suene solo por el canal derecho,
# en este caso la ganancia y la duracion dependen de la tarea
def function2():
    gain = manager.task.settings.sound_gain
    sound = whitenoise_generator(
        duration=manager.task.settings.sound_duration, ramp_time=0.005
    )
    sound *= gain
    sound_device.load(None, sound)

# generamos un sonido de ruido blanco y hacemos que suene por ambos canales con
# ganancias obtenidas de la calibración
def function3():
    gain_left = manager.sound_calibration.get_sound_gain(
        speaker=0, dB=70, sound_name="whitenoise"
    )
    gain_right = manager.sound_calibration.get_sound_gain(
        speaker=1, dB=70, sound_name="whitenoise"
    )
    sound = whitenoise_generator(
        duration=manager.task.settings.sound_duration, ramp_time=0.005
    )
    sound_left = sound * gain_left
    sound_right = sound * gain_right
    sound_device.load(sound_left, sound_right)

# obtenemos un sonido a partir de un archivo wav y lo cargamos en el dispositivo de sonido
# usamos la ganancia adecuada para ese sonido segun la calibracion
# esta funcion asume que el archivo bac.wav esta en la carpeta media del proyecto
# y que es un archivo mono o stereo (1 o 2 canales)
# simpre devuelve un array para left y otro para right (en caso de mono, left = right)
def function4():
    gain_left = manager.sound_calibration.get_sound_gain(
        speaker=0, dB=70, sound_name="whitenoise"
    )
    gain_right = manager.sound_calibration.get_sound_gain(
        speaker=1, dB=70, sound_name="whitenoise"
    )
    sound_left, sound_right = sound_device.get_sound_from_wav("bac.wav")
    sound_left *= gain_left
    sound_right *= gain_right
    sound_device.load(sound_left, sound_right)


def function5():
    sound_device.play()


def function6():
    sound_device.stop()


def function7():
    window = manager.behavior_window
    duration = manager.task.settings.stimulus_duration
    fill_color = QColor("white")
    brush_color = QColor("blue")
    x_pos = 100
    y_pos = 100
    width = 300
    height = 300
    draw_function = draw_square_generator(window, duration, fill_color, brush_color, x_pos, y_pos, width, height)
    manager.behavior_window.load_draw_function(draw_function)


# def function8():
#     manager.behavior_window.load_draw_function(
#         draw_static_image, image=manager.task.settings.image_jpg
#     )


# def function9():
#     manager.behavior_window.load_draw_function(
#         draw_video, video=manager.task.settings.video
#     )

# def function10():
#     manager.behavior_window.load_draw_function(draw_bouncing_circle)


# def function11():
#     manager.behavior_window.load_draw_function(draw_triangle)


# def function12():
#     manager.behavior_window.load_draw_function(
#         draw_video, video=manager.task.settings.video
#     )

# def function13():
#     manager.behavior_window.load_draw_function(
#         draw_static_image, image=manager.task.settings.image_png
#     )

def function14():
    manager.behavior_window.start_drawing()
    # sound_device.play()
    # task.send_softcode_to_bpod(1)

def function15():
    # is it also possible to change the background color of the window
    # the background color is used in all drawing functions to clean the window
    # the change is persistent until changed again
    manager.behavior_window.background_color = QColor(100, 100, 100)


def function16():
    # to test overriding outputs
    manager.task.bpod.manual_override_output(("PWM1", 255))  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output(("PWM1", 0))  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("Valve1")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("Valve1Off")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("BNC1High")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("BNC1Low")  # funciona


def function17():
    # to test overriding inputs
    manager.task.bpod.manual_override_input("Port1In")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_input("Port1Out")  # funciona
    time.sleep(1)


def function18():
    start_time = time.time()
    sound_device.play()
    end_time = time.time()
    print("play delay: ", end_time - start_time)
