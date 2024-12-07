
***
### This repository conatins TSM using Griffin Lim's iterative algorithm
    This algorithm iteratively estimates the phase spectra of MSTFTM
    Supports several initialization methods: 'zero_phase', 'gaussian', 'sola'

### How to use:
    Check out 'demo_selective.ipynb' for use
    lsee_tsm_selective() function:
        initial: ['zero_phase', 'gaussian', 'sola'] initialization method
        frame_ranges:[list] list of tuples that conatin (start, end) frame indices 
        rate_ranges: [list] list of rates (0.5 - 2)

### Reference:
    Roucos, Salim, and Alexander Wilgus. "High quality time-scale modification for speech." ICASSP 1985.
    Griffin, Daniel, and Jae Lim. "Signal estimation from modified short-time Fourier transform." IEEE TSP 1984.
***