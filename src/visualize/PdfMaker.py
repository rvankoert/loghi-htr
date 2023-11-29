# Imports

# > Standard library

# > Local dependencies

# > Third party dependencies
import cv2
from fpdf import FPDF


class PdfMaker(FPDF):
    # A4 @ 300 dpi - 3507x2480 pix
    # A4 @ 200 dpi - 2338 x 1653 pix
    # A4 @ 150 dpi - 1753x1240 pix
    # A4 @ 72 dpi - 841x595 pix
    DPI = 150
    MM_IN_INCH = 25.4
    A4_HEIGHT = 420
    A4_WIDTH = 297
    MAX_WIDTH = 1753
    MAX_HEIGHT = 1240

    def pixels_to_mm(self, val):
        return val * self.MM_IN_INCH / self.DPI

    def resize_to_fit(self, img_filename, scale_factor=1.0):
        width, height = self.get_image_size(img_filename)
        width_scale = self.MAX_WIDTH / width
        height_scale = self.MAX_HEIGHT / height
        scale = min(width_scale, height_scale) * scale_factor
        return round(self.pixels_to_mm(scale * width)), round(self.pixels_to_mm(scale * height))

    def centre_image(self, img, title, scale_factor=1.0):
        width, height = self.resize_to_fit(img, scale_factor)

        # Calculate the width for the title cell (half of the page width)
        title_cell_width = self.A4_WIDTH / 2

        # Calculate the X-coordinate to center the title cell
        x_title_cell = (self.w - title_cell_width) / 2

        # Add some space at the top of the text cell
        y = self.get_y() + 10
        self.set_y(y)  # Set next element y, i.e. the image

        # Add title cell with yellow background and border
        self.set_fill_color(255, 255, 0)  # Yellow background
        self.set_draw_color(0, 0, 0)  # Black border
        self.set_x(x_title_cell)  # Set X-coordinate for centering
        self.set_text_color(61, 72, 73)
        self.cell(title_cell_width, 10, title, 1, 0, 'C', 1)  # Title cell with border

        # Add some space at the bottom of the text cell
        y = self.get_y() + 10
        self.set_y(y)  # Set next element y, i.e. the image

        # Swap width/height if needed based on page orientation
        x = (self.w - width) / 2
        y = self.get_y() + 5  # Position slightly above the existing image
        self.image(img, x, y, width, height)

        # Update the Y-coordinate for the next image
        self.set_y(y + height + 10)  # 10 is the spacing between images

    def get_image_size(self, img_filename):
        image = cv2.imread(img_filename)
        # get width and height
        height, width = image.shape[:2]
        return width, height

    def set_header(self):
        self.set_font("Arial", "B", 12)
        self.set_text_color(font_r, font_g, font_b)
        self.set_draw_color(font_r, font_g, font_b)
        self.cell(0, 10, (args.replace_header if args.replace_header else extract_model_name(
            args.existing_model) + " Visualization Report"),
                  align="C",
                  ln=1,
                  border=1)v