# asymmetric-cenozoic-extreme-events

Model and data analysis code from Arnscheidt and Rothman: "Asymmetry of extreme Cenozoic climate-carbon cycle events" (2021)

Python scripts are named according to the figures they produce, and the Julia file is the numerical model. 

All other files contain data needed by the code files:

* `cenogrid_data.npy` contains isotope data from [Westerhold et al. (2020)](https://doi.pangaea.de/10.1594/PANGAEA.917503), and is included here in accordance with the [CC BY 4.0 license.](https://creativecommons.org/licenses/by/4.0/) 
* `la04...` files contain insolation data from Laskar et al. (2004), computed using the provided [web interface](http://vo.imcce.fr/insola/earth/online/earth/online/index.php).
* `ens...csv` are output data from `model.jl`.

Please reach out to me if you encounter any issues.
