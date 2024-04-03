# Imports

# > Standard library
import re

# > Local dependencies
from vis_arg_parser import get_args

# > Third party dependencies
import cv2
from fpdf import FPDF


class PdfMaker(FPDF):
    """
    Custom PDF generator class based on FPDF for creating visualization
    reports.

    Attributes:
        dpi (int): Dots per inch for the PDF document.
        mm_in_inch (float): Millimeters in an inch.
        a4_height (int): Height of an A4 sheet in millimeters.
        a4_width (int): Width of an A4 sheet in millimeters.
        max_width (int): Maximum width for images in the document.
        max_height (int): Maximum height for images in the document.
        args (Namespace): Command-line arguments obtained using the get_args
            function.
        model_name (str): Extracted model name from the existing model path.

    Methods:
        pixels_to_mm(val: int) -> float:
            Convert pixels to millimeters.

        resize_to_fit(img_filename: str, scale_factor: float = 1.0) -> tuple:
            Resize the image to fit within the specified maximum dimensions.

        centre_image(img: str, title: str, scale_factor: float = 1.0):
            Center an image in the document with a title cell above it.

        get_image_size(img_filename: str) -> tuple:
            Get the size (width, height) of the given image file.

        set_header(replace_header: bool, font_r: int, font_g: int, font_b: int):
            Set the header for the PDF document and the background/text colors.
    """

    # A4 @ 300 dpi - 3507x2480 pix
    # A4 @ 200 dpi - 2338 x 1653 pix
    # A4 @ 150 dpi - 1753x1240 pix
    # A4 @ 72 dpi - 841x595 pix
    def __init__(self):
        super().__init__(orientation="L", format="A3")
        self.dpi = 150
        self.mm_in_inch = 25.4
        self.a4_height = 420
        self.a4_width = 297
        self.max_width = 1753
        self.max_height = 1240
        self.args = get_args()
        self.model_name = self.extract_model_name(self.args.existing_model)

    def pixels_to_mm(self, val: int) -> float:
        """
        Convert pixels to millimeters.

        Args:
            val (int): Value in pixels.

        Returns:
            float: Equivalent value in millimeters.
        """
        return val * self.mm_in_inch / self.dpi

    def resize_to_fit(self, img_filename: str, scale_factor: float = 1.0) \
            -> tuple:
        """
        Resize the image to fit within the specified maximum dimensions.

        Args:
            img_filename (str): Path to the image file.
            scale_factor (float, optional): Scaling factor. Defaults to 1.0.

        Returns:
            tuple: Resized width and height in millimeters.
        """
        width, height = self.get_image_size(img_filename)
        width_scale = self.max_width / width
        height_scale = self.max_height / height
        scale = min(width_scale, height_scale) * scale_factor
        return round(self.pixels_to_mm(scale * width)), \
            round(self.pixels_to_mm(scale * height))

    def centre_image(self, img: str, title: str, scale_factor: float = 1.0):
        """
        Center an image in the document with a title cell above it.

        Args:
            img (str): Path to the image file.
            title (str): Title for the image.
            scale_factor (float, optional): Scaling factor. Defaults to 1.0.
        """

        width, height = self.resize_to_fit(img, scale_factor)

        # Calculate the width for the title cell (half of the page width)
        title_cell_width = self.a4_width / 2

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
        self.cell(title_cell_width, 10, title, 1, 0,
                  'C', 1)  # Title cell with border

        # Add some space at the bottom of the text cell
        y = self.get_y() + 10
        self.set_y(y)  # Set next element y, i.e. the image

        # Swap width/height if needed based on page orientation
        x = (self.w - width) / 2
        y = self.get_y() + 5  # Position slightly above the existing image
        self.image(img, x, y, width, height)

        # Update the Y-coordinate for the next image
        self.set_y(y + height + 10)  # 10 is the spacing between images

    def get_image_size(self, img_filename: str) -> tuple:
        """
        Get the size (width, height) of the given image file.

        Args:
            img_filename (str): Path to the image file.

        Returns:
            tuple: Image width and height in pixels.
        """

        image = cv2.imread(img_filename)
        # get width and height
        height, width = image.shape[:2]
        return width, height

    def extract_model_name(self, input_string):
        """
        Extract the model name from the given input string.

        Args:
            input_string (str): Input string containing the model name.

        Returns:
            str: Extracted model name.
        """
        # Remove trailing slashes (if any)
        cleaned_string = input_string.rstrip('/')

        # Match the last word after the last slash
        match = re.search(r'/([^/]+)$', cleaned_string)

        if match:
            return match.group(1)
        # If no match found, return the entire cleaned string
        return cleaned_string

    def set_header(self, replace_header, font_r, font_g, font_b):
        """
        Set the header for the PDF document and the background/text colors.

        Args:
            replace_header: str
                Replacement header text (if any).
            font_r: int
                Red component of the font color.
            font_g: int
                Green component of the font color.
            font_b: int
                Blue component of the font color.
        """
        self.set_font("Arial", "B", 12)
        self.set_text_color(font_r, font_g, font_b)
        self.set_draw_color(font_r, font_g, font_b)
        self.cell(0, 10, (replace_header if replace_header
                          else self.model_name + " Visualization Report"),
                  align="C",
                  ln=1,
                  border=1)
