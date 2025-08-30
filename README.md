# Plotting an equirectangular SVG onto stereographic projection

And by plotting, I mean both converting vector coordinates and tracing the result on a piece of paper. Because it all starts with a pen plotter and a love for mini planets.

*   a Blender script to batch-export a panorama composed of several freestyle SVG shots from a given viewpoint;
*   a script to remap the resulting SVGs together in a single equirectangular SVG;
*   a program that takes an equirectangular SVG and remaps the lines/paths in it to stereographic projection.

The first tool, cube\_map\_svg.py, is the most straightforward, but potentially the one that can set you up for success or failure downstream. Open your scene in Blender and position your camera where you want it. Configure your camera for a 90-degree FOV. Make sure your camera sensor fit is set to have the same width and height. I usually leave it to 36mm by 36mm. Set the output resolution to 2048×2048, though any square ratio will do. In the script, configure the output directory and keep everything else as is. Don’t forget to enable freestyle output and freestyle SVG in your render panel. Run the script from Blender.

reproj\_svg\_to\_equirect.py will take as input the metadata file generated in your export directory and process each SVG, reprojecting them all into one equirectangular SVG file. When this project was not enhanced by coding agents, I was following the wrong approach and attempting to stitch different SVGs together. It did not go well. GPT-5 largest contribution to me so far was providing the math to project the cube-map into an equirectangular format.

Once you have an equirectangular panorama, you can go crazy with the reprojections. The only one I implemented so far is the stereographic one. svg\_stereographic.py helps you make those transformations, taking parameters such as horizontal camera panning, and zoom level.

Here’s some early results using this stack:

![](https://jamez.it/blog/wp-content/uploads/2025/08/new_paris_simple_stereo3.svg)

![](https://jamez.it/blog/wp-content/uploads/2025/08/nyc3_mp_resize2.svg)

![](https://jamez.it/blog/wp-content/uploads/2025/08/body_simple2.svg)

![](https://jamez.it/blog/wp-content/uploads/2025/08/new_rome_stereo2.svg)

Let the penplotting season begin…
