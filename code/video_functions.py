from PyQt5.QtCore import QRect, Qt
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QPolygonF





def draw_square_generator(window, duration, fill_color, brush_color, x_pos, y_pos, width, height) -> callable:

    def draw():
        with QPainter(window) as painter:
            # no transparency and no antialiasing (drawing is faster)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.setRenderHint(QPainter.Antialiasing, False)

            # clean the window with background color
            painter.fillRect(painter.viewport(), window.background_color)

            if window.elapsed_time < duration:
                painter.setPen(fill_color)
                painter.setBrush(brush_color)
                painter.drawRect(x_pos, y_pos, width, height)

    return draw


# def draw_square_with_task_parameters() -> None:
#     # get the current task settings and the window
#     task = manager.task
#     window = manager.behavior_window

#     # draw a square with parameters from task
#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with the background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # draw the square only if within stimulus duration with stimulus color
#         if window.elapsed_time < task.settings.stimulus_duration:
#             painter.setPen(Qt.NoPen)
#             painter.setBrush(QColor(task.settings.color))
#             painter.drawRect(
#                 int(task.settings.x_position),
#                 int(task.settings.y_position),
#                 int(task.settings.width),
#                 int(task.settings.height),
#             )  # IMPORTANT: drawRect expects integers


# def draw_circle() -> None:
#     # get the window
#     window = manager.behavior_window

#     # draw a white circle in the center of the window with diameter 100
#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # get window size
#         width = painter.viewport().width()
#         height = painter.viewport().height()

#         # calculate position
#         diameter = 100
#         x = (width - diameter) // 2
#         y = (height - diameter) // 2

#         # set border and fill and draw the circle
#         painter.setPen(QColor("#FF0000"))
#         painter.setBrush(QColor("#FF0000"))
#         painter.drawEllipse(QRect(x, y, diameter, diameter))


# def draw_bouncing_circle() -> None:
#     # get the window
#     window = manager.behavior_window

#     # we can change the background color when drawing a stimulus
#     # the change is persistent until changed again
#     window.background_color = QColor("purple")

#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with bakground color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # get window size
#         width = painter.viewport().width()
#         height = painter.viewport().height()

#         # calculate diameter based on elapsed time
#         diameter = int(50 + abs(((window.elapsed_time * 1000) % 1500) - 750) * 0.1)

#         # calculate position
#         x = (width - diameter) // 2
#         y = (height - diameter) // 2

#         # set border and fill and draw the circle
#         painter.setPen(Qt.NoPen)
#         painter.setBrush(QColor(255, 255, 0))  # rgb color
#         painter.drawEllipse(QRect(x, y, diameter, diameter))


# def draw_triangle() -> None:
#     # get the current task and the window
#     task = manager.task
#     window = manager.behavior_window

#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # draw the square only if within stimulus duration
#         if window.elapsed_time < task.settings.stimulus_duration:
#             # get window size
#             width = painter.viewport().width()
#             height = painter.viewport().height()

#             # border and fill
#             painter.setPen(Qt.NoPen)
#             painter.setBrush(QColor(task.settings.color))

#             # draw the triangle
#             points = QPolygonF(
#                 [
#                     QPointF(width / 2, height / 2 - 100),
#                     QPointF(width / 2 - 100, height / 2 + 100),
#                     QPointF(width / 2 + 100, height / 2 + 100),
#                 ]
#             )
#             painter.drawPolygon(points)


# def draw_static_image() -> None:
#     # get the current task and the window
#     task = manager.task
#     window = manager.behavior_window

#     # draw an image with parameters from task
#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with the background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # draw the image only if within stimulus duration
#         if window.elapsed_time < task.settings.stimulus_duration:
#             painter.drawPixmap(0, 0, window.image)


# def draw_static_image_using_alpha() -> None:
#     # get the window
#     window = manager.behavior_window

#     # draw an image with parameters from task
#     with QPainter(window) as painter:
#         # in this case we use transparency as the image is a png with alpha channel
#         # CompositionMode_SourceOver instead of CompositionMode_Source
#         painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with the background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # draw the image only
#         painter.drawPixmap(0, 0, window.image)

# def draw_video() -> None:
#     # get the window
#     window = manager.behavior_window

#     # draw the last image from the video source
#     with QPainter(window) as painter:
#         # no transparency and no antialiasing (drawing is faster)
#         painter.setCompositionMode(QPainter.CompositionMode_Source)
#         painter.setRenderHint(QPainter.Antialiasing, False)

#         # clean the window with the background color
#         painter.fillRect(painter.viewport(), window.background_color)

#         # get the last frame from the video source
#         image = window.get_video_frame()

#         # draw the last frame from the video source
#         if image is not None:
#             painter.drawImage(0, 0, image)
