# 🔍 FFT Pattern Analyzer with Wallpaper Group Detection

A Python GUI app for batch-processing JPG/PNG images using 2D Fourier Transform (FFT), enhanced with:
- Automatic wallpaper group analysis
- Custom color palette generation based on image content
- Multi-zoom FFT visualizations
- Clean labeled outputs in a separate folder

---

## 🎯 Features

✅ Batch process all JPG/PNG images in a selected folder  
  * Does all the images in the folder targeted by the GUI
🌀 Extract & visualize 2D Fourier Transforms (FFT)  

🎨 Use dominant image colors to build matching colormaps  
 * The off the peg colour schemes were all rubbish so I asked chat GPT to cook up a custom scheme for each FFT based on the main colours in the image. This feature is killer.
🔍 Save FFTs at multiple zoom levels (25%, 50%, 100%)  
 * It belches out files for all four levels of zoom - choose the one that looks coolest for your image, soz you have to delete the others.
ICONHERE Apply three contrast methods: log, gamma, stretch
  * The outputs were a bit flat and noisy so I asked chatGPT to do some maths thing to make them pop a bit more - again all versions of the image are belched out.
🧠 Heuristic-based wallpaper group classification  
  * Really crude wallpaper group determination with very dubious outcomes (doesn't consider all 17 wall paper groups, because maths is hard). Does this bit by analysing the FFT and looking for symmetry. Sometimes spots chirality! Assigns lots of random crap as P4m.
📑 Adds group name as label under each FFT image  
  * Instagram or CrystEngComm Ready!
💾 Saves outputs into a clean `output_fft/` folder  
  * Great for rapid deletion of all generated files after fuckups.
🖼 GUI built with Tkinter — easy to use!
  * Did I mention I'm not the command line type.
---

## 📦 Requirements

Install the needed libraries:

```bash
pip install numpy matplotlib pillow scikit-learn opencv-python
