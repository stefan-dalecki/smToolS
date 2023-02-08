![smToolS](https://user-images.githubusercontent.com/89150776/184039244-0408a008-0f11-4c65-a6f5-ae7cab6031d1.png)

# Cześć!  

and welcome to a series of tools created to help scientists analyze single molecule movies.  
While much of this software is specific to bio-physical research, significant effort has been made keep these tools scalable.  
This work is, and will always be, public. Proper science benefits society, not the individual.  
Nonetheless, if you use this tool, please take time to mention Stefan Dalecki and the Falke Lab at CU Boulder.

# Setup
This repo uses Python 3.11. To create a python virtual environment for this repository and install requirements, use the below commands.
```bash
python -m venv venv
pip install -r requirements.txt
```

# Running Scripts
This repo utilizes the python parser library. Necessary details to run all scripts must be supplied at runtime. While there are some base parsers, each file within `.\smToolS\scripts` is unique. Use the `--help` argument for additional details when trying to run a script.
For example:
```bash
python .\smTools\scripts\analyze.py --help
```

# analyze.py

`analyze.py`, aka the image processing toolkit utilizes three main particle tracking softwares to detect particle diffusion on a 2-dimensional surface.  
All three return data related to particle movement, with certain Trackmate '.xml' files lacking brightness information. 
Within smToolS, trackpy linking is drawn from an '.nd2' set of frames.
Below is a short summary of each method with their respective program and output filetype used in this toolbox.  

1. ParticleTrackerClassic : an ImageJ/FIJI Plugin : '.csv'
2. Trackmate : another ImageJ/FIJI Plugin : '.xml'
3. Trackpy : Python specific particle tracker : '.h5'

Once read, trajectory information is later filtered, modeled, and analyzed, with specific criterion for each mentioned feature toggled upon program startup.


# visualize.py

|In theory|In practice|
|---|---|
|A crucial part of trajectory evaluation is setting filter values to only capture 'protein molecules' (or any foreground piece) from your movies. By using the 'Visualization' tool, preliminary steps can be taken to identify various groups of molecules within your already tracked movie data. One can test various thresholding cutoffs and immediately understand which trajectories are removed and why.|![biggest](https://user-images.githubusercontent.com/89150776/184054184-067ad310-d4af-4f92-baec-6f36b0ecb39a.png)


# Machine Learning

Most data used in the 'Image Processing' directory contains only average brightness, length, and diffusion metrics.  
However, this computational tool contains a prebuilt foundation for analyzing exceeding multidimensional challenges  
where in addition to the mentioned metrics, shape and brightness variance are considered.  
Importantly, machine learning offers a degree of generalizability that is absent in strict filter cutoffs,  
allowing for valid outlier trajectories to be kept in exceptions where another filter would incidentally remove them.

## NOTE
until TensorFlow is supported on Python 3.11, it is recommended that you use Python 3.10 and manually install the package.
```bash
pip install tensorflow
```

# Support

I, Stefan Dalecki, will provide limited support of this repository from 08/22/2022 onward.  
It is yet to be determined whether another will take over my role in this project.  
If you are interested in building upon this work, contact Dr. Joseph Falke of CU Boulder.  
