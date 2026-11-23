# Optical network models

The five-node ring model compares routes by distance, signal-to-noise ratio and capacity. It assigns the same wavelength on every link of an accepted route and shows how the 20 directed node-to-node requests occupy the ring. The 14-node NSF model finds shortest routes by fibre length.

## Run the models

Use Python 3.10 or later. Run these commands from the repository root.

~~~sh
python -m pip install -r requirements.txt
python -m optical.step01_ring_topology --no-show
python -m optical.nsf_topology --no-show
python -m unittest optical.test_ring_capacity
~~~

## Results and calculations

- [Five-node ring results](optical/results_ring_step01/)
- [NSF route examples](optical/results_nsf/)
- [400 km optical link parameter validation (PDF)](optical/Engineering%20Validation%20of%20400%20km%20Optical%20Link%20SNR%20Parameters.pdf)
- [Handwritten 400 km SNR calculation (PDF)](optical/hand-written%20calculation%20of%20400km%20SNR.pdf)

The ring uses the [physical-layer model](reference_physical_model.py) to calculate its 400 km reference operating point.
