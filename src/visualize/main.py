# Imports

# > Standard library
import logging

# > Local dependencies
import visualize_filters_activations
import visualize_timestep_predictions
from PdfMaker import PdfMaker
from vis_arg_parser import get_args

# > Third party dependencies

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting visualization processes...")
    args = get_args()
    dictionary = args.__dict__
    print(dictionary)
    logger.info("Starting filter activation visualizer")
    visualize_filters_activations.main(args)
    logger.info("Starting timestep prediction visualizer")
    visualize_timestep_predictions.main(args)

    # Add titles (you can customize this part)
    png_title = "Filter activations"
    jpg_title = "Example predictions per timestep"

    # Initiate PDF
    pdf = PdfMaker()
    pdf.add_page()

    # Set color_scheme
    if args.light_mode:
        font_r, font_g, font_b = 0, 0, 0
    else:
        font_r, font_g, font_b = 255, 255, 255
        pdf.rect(0, 0, pdf.w, pdf.h,
                 'F')  # Draw a rectangle to fill the entire page with the default background color
    pdf.set_header(args.replace_header, font_r, font_g, font_b)

    ts_plot = ("visualize_plots/timestep_prediction_plot"
               + ("_light" if args.light_mode else "_dark")
               + ".jpg")
    act_plot = ("visualize_plots/model_new10_1channel_filters_act"
                + ("_light" if args.light_mode else "_dark")
                + ("_detailed" if args.do_detailed else "")
                + ".png")

    pdf.centre_image(ts_plot, "Example prediction for sample image: ", scale_factor=0.5)
    pdf.centre_image(act_plot, "Learned Conv Filters & Filter Activations", scale_factor=1.25)

    # Save the PDF
    pdf.output("visualize_plots/"
               + (args.replace_header if args.replace_header else pdf.extract_model_name(args.existing_model))
               + "_visualization_report"
               + ("_light" if args.light_mode else "_dark")
               + ".pdf")