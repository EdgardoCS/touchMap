# TOUCH MAP 

This project performs a pixelwise comparison between two or more images using statistical testing. Specifically, it conducts a t-test for each corresponding pixel across images and applies FDR (False Discovery Rate) correction to control for multiple comparisons. The results are visualized as heatmaps for easier interpretation.

🔍 What It Does

    Takes two or more input images of the same dimensions.
    Performs a t-test on each pixel across the image stack.
    Applies FDR correction to adjust p-values.
    Outputs:

        T-map: visual representation of the t-values.
        P-map: heatmap highlighting statistically significant pixels.

📦 Features

    Handles any number of input images.
    Statistical rigor using pixel-by-pixel inference.
    Visualization of both statistical strength and significance.

🧠 Ideal For

    Image processing and analysis.
    Scientific imaging and fMRI