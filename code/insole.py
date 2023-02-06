import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy import ndimage, stats
import math
from scipy import interpolate
import scipy
from math import ceil, sqrt
from scipy import signal
import bz2
import _pickle as cPickle
import footsim as fs
from skimage.transform import resize
from copy import deepcopy
from scipy import spatial

def get_foot_outline():
    """ Get the x and y coordinates relating the boundaries of the foot used within mapping

    Returns:
        x_outline (np.array): x coordinates
        y_outline (np.array): y coordinates

    """
    # Get foot boundary coordinates
    boundaries = fs.foot_surface.boundaries

    # turn coordinates from pixel space to foot space
    x_outline = []
    y_outline = []
    for i in range(len(boundaries)):
        for j in range(len(boundaries[i])):
            locs = fs.foot_surface.pixel2hand(np.array(boundaries[i][j]))
            x_outline = np.append(x_outline, locs[0])
            y_outline = np.append(y_outline, locs[1])

    return x_outline, y_outline

def import_data(filepath, **args):
    """ Reads a *.csv file with empirically recorded datapoints from a Tekscan Pressure Measurement System 7.00-22

    * UNITS KPa
    * SECONDS_PER_FRAME 0.01 (Per frame: ROWS 60 COLS 21)

    Args:
        filepath: filepath (str): path to the *.csv file
        **args:
            calibration_type (str): name of calibration type conducted
            extended_calibration (bool): True if extended calibration occurred, False if not
            frames (float): number of frames saved

    Returns:
        D (np.array()): 3D matrix of frames recorded

    """

    calibration_type = args.get('calibration_type', 'step')
    extended_calibration = args.get('extended_calibration', False)
    frames = args.get('frames', 2000)

    # Skips the heading of the data file and uses 61 lines per frame #
    if calibration_type == 'point':

        if extended_calibration == True:
            CSV = pd.read_csv(filepath, skiprows=34, header=None, nrows=frames * 61) # read in .csv file
            D_tmp = CSV.values
            D = D_tmp.reshape((-1, 61, 21,)).transpose(1, 2, 0)  # 3D matrix

        else:
            CSV = pd.read_csv(filepath, skiprows=33, header=None, nrows=frames * 61)
            D_tmp = CSV.values
            D = D_tmp.reshape((-1, 61, 21,)).transpose(1, 2, 0)  # 3D matrix


    else:
        CSV = pd.read_csv(filepath, skiprows=33, header=None,nrows=frames * 61)  # skiprows has to be different sometimes
        D_tmp = CSV.values
        D = D_tmp.reshape((-1, 61, 21,)).transpose(1, 2, 0)  # 3D matrix

    # cut off frame headers and convert to float

    D = D[0:-2, :, :]
    D[D == 'B'] = np.nan
    D = D.astype(float)

    return D

def spatial_filter(D, **args):
    """ Apply a Gaussian filter across the spatial dimension of pressure data

    Args:
        D (np.array): Raw pressure data
        **args:
            sigma (float): Gaussian filter strength

    Returns:
        Z (np.array): filtered matrix
    """

    sigma = args.get('sigma', .5)

    U=D

    V=U.copy()
    V[np.isnan(U)]=0 # turn NaN values to 0.
    VV=scipy.ndimage.gaussian_filter(V,sigma=sigma) # apply filter

    W=0*U.copy()+1
    W[np.isnan(U)]=0 # turn NaN values to 0
    WW=scipy.ndimage.gaussian_filter(W,sigma=sigma) # apply filter

    Z=VV/WW

    return Z

def temporal_filter(vector):
    """ Apply a low-pass butterworth filter to filter the data temporally

    :param
        vector (np.array): a 1D array of pressure
    :return:
        filtered (np.array): filtered data
    """
    fc = 18  # Cut-off frequency of the filter (18Hz)
    w = fc / (100 / 2)  # Normalize the frequency
    b, a = signal.butter(1, w, 'low')
    filtered = signal.filtfilt(b, a, vector)
    filtered[filtered < 0] = 0  # ensures no negative pressures

    return filtered

def filter_data(D):
    """ Apply the spatial and temporal filters to the raw data

    Args:
        D (np.array): Raw pressure data

    Returns:
        D (np.array): filtered data

    """

    # apply spatial filter to data
    D = apply_spatial_filter(D)

    # apply temporal filter to data
    D = apply_temporal_filter(D)

    return D

def apply_spatial_filter(D):
    """ Apply spatial Gaussian filter to each frame in the pressure data

    Args:
        D (np.array): Raw pressure data

    Returns:
        D (np.array): filtered pressure matrix
    """

    # loop through each frame in the pressure data
    for i in range(D.shape[2]):

        # filter spatially
        D[:,:,i] = spatial_filter(D[:,:,i])

    return D

def apply_temporal_filter(D):
    """ Apply Butterworth filter to temporal aspect of the data

    Args:
        D (np.array): Raw data

    Returns:
        D (np.array): filtered data

    """

    # loop through y axis of data
    for i in range(D.shape[1]):

        # loop through x axis of data
        for j in range(D.shape[0]):

            # apply butterworth filter to data
            D[j, i, :] = temporal_filter(D[j, i, :])

    return D

def flip_matrix(D):
    """ Flips right foot matrix in order for the matrix to work in FootSim, will works on the left foot

    Args:
        D (np.array): Pressure data from the insoles

    Returns:
        D (np.array): flipped matrix

    """

    # loop through the time dimension of the pressure matrix
    for i in range(D.shape[2]):

        # flip matrix along the x axis
        D[:,:,i] = np.flip(D[:,:,i],axis=1)

    return D

def create_calibration_data(D1, D2, **args):
    """ Takes data from multiple trials and concatenates into one np.array. Requires at least 2 matrices to concatenate

    Args:
        D1 (np.array): Matrix of pressure data
        D2 (np.array): Matrix of pressure data
        **args:
            D3 (np.array): Matrix of pressure data
            D4 (np.array): Matrix of pressure data
            D5 (np.array): Matrix of pressure data

    Returns:
        D (np.array): concatenated matrix
    """

    D3 = args.get('D3', np.zeros(D1.shape))
    D4 = args.get('D4', np.zeros(D1.shape))
    D5 = args.get('D5', np.zeros(D1.shape))

    # concatenate data
    D = np.concatenate((D1,D2,D3,D4,D5), axis=2)

    return D

def cut_frame(D, calibrate, **args):
    """ Remove outer borders from pressure matrix

    Args:
        D (np.array): input data
        calibrate (bool): if True, will cut frame based on values within array. If False, will use pre-determined parameters entered through **args
        **args:
            min1 (int): index along the x axis where pressure starts
            min2 (int): index along the x axis where pressure finishes
            max1 (int): index along the y axis where pressure starts
            max2 (int): index along the y axis where pressure finishes

    Returns:

    """

    min2 = args.get('min2')
    max2 = args.get('max2')
    min1 = args.get('min1')
    max1 = args.get('max1')

    if calibrate == True:

        Dm = np.nanmean(D, axis=2)
        a1 = np.nanmax(Dm, axis=0)
        idx1 = np.where(a1 > 1) # find where pressure values are greater than 1
        if len(idx1[0]) != 0: # checks if pressure reaches the outer boundary of the matrix along the x axis
            min1 = np.min(idx1)
            max1 = np.max(idx1)
        else: # if pressure reaches the outer boundary of the matrix, use default min and max to cut data
            min1 = 0
            max1 = 20

        a2 = np.nanmax(Dm, axis=1)
        # find indexes where pressure value is greater than zero
        idx2 = np.where(a2 > 1)
        if len(idx2[0]) != 0: # checks if max pressure reaches the outer boundary of the matrix along the y axis
            min2 = np.min(idx2)
            max2 = np.max(idx2)
        else: # if max pressure reaches the outer boundary of the matrix, use default min and max to cut data
            min2 = 0
            max2 = 20

        # cut data and provide values to use to allow for consistent cutting for future data
        return D[min2:max2 + 1, min1:max1 + 1, :], min2, max2, min1, max1

    else:

        return D[min2:max2 + 1, min1:max1 + 1, :] # cut data

def map2footsim(D):
    """ Maps the pressures empirically recorded from a Tekscan Pressure Measurement System 7.00-22 into FootSim stimuli
    by changing the trace.

    Args:
        D (np.array): Pressure data processed with import_data() and cut_frame()

    Returns:
        s (FootSim Stimulus Object): contains sensor locations within s.locations
        regions (list): list of regions that relate to the sensor locations
        reshaped_data (2D np.array()): array containing only information from sensors that are used. Shape = (number of sensors, number of timepoints)
        idxs (list): list of integers relating to sensors mapped onto the foot
        D (np.array()): pressure matrix

    """

    dim = D.shape  # Dimensions of pressure data

    # get outline of the model foot
    x_outline, y_outline = get_foot_outline()

    cmin = [min(x_outline), min(y_outline)]
    cmax = [max(x_outline), max(y_outline)]

    # generate array of equally spaced coordinates between the minimum and maximum of the foot outline
    c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    c1 = np.linspace(cmin[0], cmax[0], dim[1])

    #dim = D.shape  # Dimensions
    #cmin = np.min(fs.foot_surface.bbox_min, axis=0)  # Calculates bounding box for arbitrary boundary. # y
    #cmax = np.max(fs.foot_surface.bbox_max, axis=0)  # x

    #c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    #c1 = np.linspace(cmin[0], cmax[0], dim[1])


    loc = np.zeros((0, 2)) # array to store sensor locations
    trace = np.zeros((0, dim[2])) # array to store pressure traces

    regions = []
    reshaped_data = np.zeros((0, dim[2]))
    m = 0
    idxs = []
    # loop through each point in the pressure matrix
    for i in range(dim[1]):

        for j in range(dim[0]):
            m += 1

            # coordinate of the sensor
            loca = np.array([c1[i], c0[j]])
            #loca = fs.foot_surface.pixel2hand(np.array([c1[i], c0[j]]))

            # identify region of the foot the sensor is located in
            region = fs.foot_surface.locate(loca)

            if region[0][0] == None:
                continue

            if region[0][0] == '':
                # if the location is not on the foot, set values to zero
                D[j, i, :] = 0.0

            else:

                if np.isnan(D[j, i, 0]) or np.max(D[j, i, :]) == 0.0:
                    continue

                else:
                    # keep track of those locations in the pressure matrix that are mapped onto the foot
                    idxs.append(m - 1)

                    regions = np.append(regions, region[0][0]) # append region to list

                    # store pressures for all regions on the foot only
                    reshaped_data = np.vstack((reshaped_data, D[j, i, :]))

                    # save sensor location
                    loc = np.vstack((loc, loca))

                    # D is the 21 (columns per frame) x 61 (lines per frame) x 1794 (frames) 3D matrix from the dataset

                    # new calculation of indentation, using pressure, Young's modulus and Poisson's ratio
                    indent = fs.transduction.indentation(region=region[0][0], pressure=D[j, i, :].flatten())

                    # trace = np.vstack((trace, D[j, i, :].flatten()/129.0))
                    trace = np.vstack((trace, indent))

    # generate stimulus location
    s = fs.Stimulus(location=loc, trace=trace, fs=100, pin_radius=2.5)  # SENSEL_AREA 0.258064 cm2

    return s, regions, reshaped_data, idxs, D

def map_given_locs(D, idxs):
    """ Maps the pressures empirically recorded from a Tekscan Pressure Measurement System 7.00-22 into FootSim stimuli
    by changing the trace. This version allows for stimuli to be placed in the same location as another stimulus object.
    * Requires D to be the same shape as the initial stimulus
    * Requires map2footsim to have been run on the initial stimulus

    Args:
        D (np.array()): Pressure data processed with import_data() and cut_frame()
        idxs (list): list of indicies that refer to the locations that a stimulus will be placed - generated using map2footsim

    Returns:
        s (FootSim Stimulus Object): contains sensor locations within s.locations
        regions (list): list of regions that relate to the sensor locations
        reshaped_data (2D np.array()): array containing only information from sensors that are used. Shape = (number of sensors, number of timepoints)

    """


    dim = D.shape  # Dimensions
    # Get foot boundary coordinates

    x_outline, y_outline = get_foot_outline()

    cmin = [min(x_outline), min(y_outline)]
    cmax = [max(x_outline), max(y_outline)]

    c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    c1 = np.linspace(cmin[0], cmax[0], dim[1])

    #dim = D.shape  # Dimensions
    #cmin = np.min(fs.foot_surface.bbox_min, axis=0)  # Calculates bounding box for arbitrary boundary. # y
    #cmax = np.max(fs.foot_surface.bbox_max, axis=0)  # x

    #c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    #c1 = np.linspace(cmin[0], cmax[0], dim[1])

    loc = np.zeros((0, 2))
    trace = np.zeros((0, dim[2]))

    regions = []
    reshaped_data = np.zeros((0, dim[2]))
    m=0

    for i in range(dim[1]):

        for j in range(dim[0]):

            m+=1

            #loca = fs.foot_surface.pixel2hand(np.array([c1[i], c0[j]]))
            loca = np.array([c1[i], c0[j]])

            if m-1 not in idxs:
                continue

            else:
                if np.max(D[j, i, :]) == 0.0:
                    D[j, i, :] = 0.0

                region = fs.foot_surface.locate(loca)
                #region = foot_surface.locate(loca)

                regions = np.append(regions, region[0][0])

                # store pressures for all regions on the foot
                reshaped_data = np.vstack((reshaped_data, D[j, i, :]))

                loc = np.vstack((loc, loca))

                # D is the 21 (columns per frame) x 61 (lines per frame) x 1794 (frames) 3D matrix from the dataset

                # new calculation of indentation, using pressure, Young's modulus and Poisson's ratio
                indent = fs.transduction.indentation(region=region[0][0], pressure=D[j, i, :].flatten())

                # trace = np.vstack((trace, D[j, i, :].flatten()/129.0))
                trace = np.vstack((trace, indent))

    s = fs.Stimulus(location=loc, trace=trace, fs=100, pin_radius=2.5)  # SENSEL_AREA 0.258064 cm2

    return s, regions, reshaped_data

def recalibrate_data(D, mass, **args):
    """ Identify a constant to recalibrate the pressure data so the average force == participant mass

    Args:
        D (np.array()): Single leg standing pressure data
        mass (float): participant mass (kg)
        **args:
            plot (bool): if True, plot to show recalibration

    Returns:
        calibration_constant (float): constant to multiple data by.
        The absolute value of 1 - calibration constant is the error of the insoles
    """
    plot = args.get('plot', False)

    # calculate expected participant mass based on pressure data
    expected_mass = expected_participant_weight(D)

    # generate calibration constant
    calibration_constant = mass / np.nanmean(expected_mass)

    # multiple data by calibration constant
    recalibrated = D * calibration_constant


    if plot == True:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(expected_mass, label='calculated mass, average: ' + str(round(np.mean(expected_mass), 1)))
        plt.axhline(y=mass, c='r', label='actual mass: ' + str(mass))
        plt.xticks([])
        plt.ylabel('Mass (kg)')
        plt.legend()

        # calculate new expected mass
        expected_mass_recalibrated = expected_participant_weight(recalibrated)

        plt.subplot(2, 1, 2)
        plt.plot(expected_mass_recalibrated, label='calculated mass, average: ' + str(round(np.mean(expected_mass),1)))
        plt.axhline(y=mass, c='r', label='actual mass: ' + str(mass))
        plt.xticks([])
        plt.ylabel('Mass (kg)')
        plt.legend()
        plt.show()

    return recalibrated, calibration_constant

def all_frames_analysis(D, participant, foot, frames, idxs, output_path, plot):
    """ For project insole, only need pressure data, no afferent information or extended stimulus information. Only
    need the footsim stimulus locations

    :param D (np.array()): Pressure data
    :param participant (str): participant id
    :param foot (str): foot (left or right)
    :param frames (int): D.shape[2], number of samples in the data
    :param idxs (list): stimulus locations across the foot
    :param output_path (str): filepath to save data
    :param plot (bool): True or False - whether to plot during preprocessing
    :return:
        data.pbz2 - dictionary containing preprocessed data
    """

    data = {}

    data['Raw data'] = D

    # map data onto foot
    s, regions, reshaped_data = map_given_locs(D[:,:,0:5], idxs)
    data['Stimulus'] = s # only need the simulus locations for project insole, which is why only a couple of frames are needed to be passed
    data['Regions'] = regions
    reshaped_data = reshape_data(D, idxs)
    data['Reshaped data'] = reshaped_data

    # calculate total pressure
    under_threshold, step_start, index_differences, total_pressure = check4steps(D, frames=frames, output_path=output_path,plot=plot)
    data['Total pressure'] = total_pressure

    # calculate centre of pressure coordinates
    loc, y, x = centre_of_pressure(D)
    data['CoP coordinates'] = loc

    # calculate pressure per region
    av_pressure_per_region, contact_percentage_per_region, total_pressure_per_region = average_pressure_per_region(reshaped_data, regions,
                                                                                        foot=foot, \
                                                                                        participant=participant,
                                                                                        frames=frames,plot=plot,output_path=output_path)
    data['Average pressure per region'] = av_pressure_per_region
    data['Contact percentage per region'] = contact_percentage_per_region
    data['Total pressure per region'] = total_pressure_per_region

    # save data
    compressed_pickle(output_path + foot + " data", data)

    return data

def expected_participant_weight(D, **args):
    """ Calculates the expected mass of the participant

    Args:
        D (np.array()): pressure data from insoles
        **args:
            sensor_area (float) = area of a single sensor (in m^2)
            number_of_sensors (int) = number of sensors on the insole

    Returns:
        mass = weight of participant (kg)

    """
    D[D < 1.] = 0.0
    sensor_area = args.get('sensor_area', .0000258064) # m^2

    # array to store force data
    new = np.zeros(D.shape)

    mass = np.zeros(D.shape[2])

    # loop through the time dimension of the data
    for i in range(D.shape[2]):

        # multiple by 1000 to convert from kPa to N/m^2
        new[:, :, i] = D[:, :, i] * 1000

        # multiply by sensor area to get force (N)
        new[:, :, i] = new[:, :, i] * sensor_area

        # divide by gravity to convert from N to mass per sensor
        new[:, :, i] = new[:, :, i] / 9.80665

        # sum all values to get total mass
        mass[i] = np.nansum(new[:, :, i])

    return mass


def pressure_to_force(D, **args):
    """ Converts pressure from kPa to force in Newtons

    Args:
        D (np.array()): pressure data from insoles
        **args:
            sensor_area (float) = area of a single sensor
            number_of_sensors (int) = number of sensors on the insole

    Returns:
        force (np.array()): matrix containing values in Newtons

    """
    sensor_area = args.get('sensor_area', .0000258064)  # m^2

    force = np.zeros(D.shape)

    # loop through the time dimension of the data
    for i in range(D.shape[2]):

        # multiple by 1000 to convert from kPa to N/m^2
        force[:, :, i] = D[:, :, i] * 1000

        # multiply by sensor area to get force (N)
        force[:, :, i] = force[:, :, i] * sensor_area

    return force

def moving_average(x, w):
    """ Calculate a moving average of a 1D vector

    Args:
        x (np.array()): 1D array of pressure data
        w (int): window size

    Returns:
        Averaged data (np.array())
    """

    return np.convolve(x, np.ones(w), 'valid') / w

def reshape_data(D, idxs):
    """ Reshape pressure data from 3D to 4D to remove sensors not placed on the foot

    Args:
        D (np.array()): pressure matrix
        idxs (list): list of numbers relating to pressure sensors that are active

    Returns:
        reshaped (np.array()): 2D array of pressure data

    """
    reshaped = np.zeros((0, D.shape[2]))

    idx = 0

    # loop through each node in the pressure matrix
    for i in range(D.shape[1]):

        for j in range(D.shape[0]):
            idx +=1
            if idx-1 not in idxs:
                continue

            else:
                reshaped = np.vstack((reshaped, D[j,i,:]))

    return reshaped

def contact_area_per_region(D, regions, reshaped_data, **args):
    """ Calculates the area of the foot in contact with the ground at each time point, broken down into regions

    Args:
        D (np.array()): pressure matrix
        regions (list): list of sensor locations
        reshaped_data (np.array()): 2D array of pressure data
        **args:
            sensor_area (float): sensor area (in m^2)

    Returns:
        contact_areas (dict): the area of the foot in contact with the ground (cm2), broken down into regions
        contact_area_as_percent (dict): the area of the foot in contact with the ground as a percentage, broken down into regions

    """

    sensor_area = args.get('sensor_area',0.258064)

    # turn pressure values of less than 1 to 0 as this reflects noise
    reshaped_data[reshaped_data < 1] = 0.0

    # calculate maximum possible area
    maximum_possible_area = reshaped_data.shape[0] * sensor_area

    contact_areas = {}
    contact_area_as_percent = {}

    # loop through each foot region id
    for region in fs.constants.foot_tags:

        contact_areas[region] = np.zeros(reshaped_data.shape[1])
        contact_area_as_percent[region] = np.zeros(reshaped_data.shape[1])

        # loop through each sensor
        for i in range(reshaped_data.shape[1]):

            # find indexes of sensors relating to the region in question
            sensor_idx = np.where(regions == region)

            # calculate contact area per region by multiplying number of active sensors by sensor area
            contact_areas[region][i] = len(np.where(reshaped_data[sensor_idx, i] > 0)[0]) * sensor_area

            # calculate contact area as a percentage of possible of the entire foot
            contact_area_as_percent[region][i] = (contact_areas[region][i] / maximum_possible_area) * 100

    return contact_areas, contact_area_as_percent

def total_contact_area(contact_areas):
    """ Calculates the area of the foot in contact with the ground at each time point, a sum of all the regional
    contact areas

    Returns:
        (float) area of the entire foot in contact with the ground (cm2), broken down into regions

    """

    return sum(contact_areas.values())

def centre_of_pressure(D, **args):
    """ Find the centre of pressure trace throughout the data

    Args:
        D: Pressure insole data
        **args:
            threshold (int): threshold of total pressure to use to allow calculations to occur on the frame

    Returns:
        loc: co-ordinates of centre of pressure coordinates in foot space ([y, x])
        y: array containing y coordinates for centre of pressure in pixel space
        x: array containing x coordinates for centre of pressure in pixel space

    """
    threshold = args.get('threshold', 500)

    D = np.nan_to_num(D, nan=0)
    D[D < 1.] = 0.0
    dim = D.shape  # Dimensions

    x_outline, y_outline = get_foot_outline()

    cmin = [min(x_outline), min(y_outline)]
    cmax = [max(x_outline), max(y_outline)]

    c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    c1 = np.linspace(cmin[0], cmax[0], dim[1])

    #dim = D.shape  # Dimensions
    #cmin = np.min(fs.foot_surface.bbox_min, axis=0)  # Calculates bounding box for arbitrary boundary. # y
    #cmax = np.max(fs.foot_surface.bbox_max, axis=0)  # x

    #print(cmin, cmax)
    #c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    #c1 = np.linspace(cmin[0], cmax[0], dim[1])

    loc = np.zeros((0, 2))
    x = list()
    y = list()

    # loop through time dimension of data
    for i in range(dim[2]):

        if np.sum(D[:, :, i]) < threshold:
            frame_COM = [np.nan, np.nan]

            loc = np.vstack((loc, frame_COM))

            y = np.append(x, frame_COM[1])
            x = np.append(y, frame_COM[0])

        else:

            # find coordinates of centre of mass
            frame_COM = ndimage.center_of_mass(D[:, :, i])
            frame_COM = np.nan_to_num(frame_COM, nan=0.)


            diff_one_x = c1[math.ceil(frame_COM[1])] - c1[math.floor(frame_COM[1])]
            diff_two_x = frame_COM[1] - int(frame_COM[1])
            add_x = 1 / diff_two_x
            ans_x = c1[math.floor(frame_COM[1])] + (diff_one_x/add_x)

            diff_one_y = c0[math.ceil(frame_COM[0])] - c0[math.floor(frame_COM[0])]
            diff_two_y = frame_COM[0] - int(frame_COM[0])
            add_y = 1 / diff_two_y
            ans_y = c0[math.floor(frame_COM[0])] + (diff_one_y / add_y)

            x = ans_x
            y = ans_y

            # transform coordinates into foot space
            loca = np.array([ans_x, ans_y])
            #loca = fs.foot_surface.pixel2hand(np.array([ans_x, ans_y]))

            #print(loca)
            loc = np.vstack((loc, loca))

            y = np.append(x, frame_COM[1])
            x = np.append(y, frame_COM[0])

    return loc, y, x

def segment_pressure_data(total_pressure, segmentation_file_name, accel_data_filename, stomp, **args):
    """ Segment pressure to remove turns from the data and plot

    Args:
        total_pressure (np.array()): array contianing total pressure
        segmentation_file_name (str): filepath to a file containing timepoints that a turn starts and ends
        accel_data_filename (str): filepath to file  containing IMU acceleration
        stomp (int): timepoint at which a stomp occurred (used to sync pressure and IMU data)
        **args:
            plot (bool): if True, plot total pressure with vertical lines demonstrating when turns occurred

    Returns:

    """

    plot = args.get('plot', False)
    if plot == True:
        output_path = args.get('output_path')

    # read in segementation times
    segmentation_times = np.asarray(pd.read_csv(segmentation_file_name, header=None)).T[0]

    IMU_data = np.asarray(pd.read_csv(accel_data_filename)).T ## <-- ACCELERATION

    # identify stomp from IMU data
    IMU_stomp = read_left_foot_accel_csv(accel_data_filename)

    pressure_stomp = stomp

    total_pressure = total_pressure[pressure_stomp:]

    segmentation_times = segmentation_times - IMU_stomp

    if plot == True:
        fig, ax = plt.subplots()
        plt.title('Pressure and IMU acceleration data - left foot')
        ax.plot(total_pressure, c='b')
        ax.set_ylabel('total pressure (kPa)')
        ax2 = ax.twinx()
        ax2.plot(IMU_data[2][IMU_stomp:], c='coral') # plot acceleration
        #ax2.plot(gyro[2][IMU_stomp:], c='coral')
        ax2.set_ylabel('IMU Gyro data')
        #plt.savefig(output_path + 'Overlaying of pressure and IMU data')

    if plot == True:
        plt.figure()
        plt.title('Segmentations pressure data')
        plt.plot(total_pressure)
        plt.ylabel('Total pressure (kPa)')
        for i in range(len(segmentation_times)):
            plt.axvline(segmentation_times[i], c='r', lw=1)
        plt.xticks([])
        plt.show()
        plt.savefig(output_path + 'Segmented pressure data.png')
        #plt.show()

def find_turning_idxs(segmentaion_file_path, gryo_data_filename):
    """ Identify timepoints during which a turn occurs

    Args:
        segmentation_file_name (str): filepath to a file containing timepoints that a turn starts and ends
        gryo_data_filename (str):  filepath to a file containing gyroscope data from the IMU

    Returns:
        turn_idxs (np.array()): array containing timepoints during which turns occurred

    """

    # read in segementation times
    segmentation_times = np.asarray(pd.read_csv(segmentaion_file_path, header=None)).T[0]

    IMU_data = np.asarray(pd.read_csv(gryo_data_filename)).T

    IMU_stomp = read_left_foot_accel_csv(gryo_data_filename)

    segmentation_times = segmentation_times - IMU_stomp

    # find maximum difference in turn segmentation times
    differences = []
    for i in range(int(len(segmentation_times) / 2)):
        differences.append(segmentation_times[(i * 2) + 1] - segmentation_times[i * 2])

    turn_idxs = np.zeros(((int(len(segmentation_times) / 2)), np.max(differences)))
    for i in range(int(len(segmentation_times) / 2)):
        idxs = np.arange(segmentation_times[i * 2], [segmentation_times[(i * 2) + 1]])

        turn_idxs[i, :len(idxs)] = idxs

    return turn_idxs

def average_after_exclusions(D, foot, step_start, under_threshold, index_differences, turn_idxs, **args):
    """ Calculate the average pressure applied during a step after removing steps during which the particpant was turning

    Args:
        D (np.array()): Pressure data - must have already had the stomp set to zero - to align with IMU signal
        foot (str): foot (left or right)
        step_start (np.array()): indexes of when each step begins
        under_threshold (np.array): indexes of frames where total pressure was below a specified threshold
                (assuming this is when the foot was off the floor)
        index_differences: difference between two indexes. when larger than 1, it is assumed a step has taken place
        turn_idxs (np.array()): array containing timepoints during which turns occurred
        **args:
            participant = participant code
            frames (int) = number of frames to sample overall
            buffer (int) = number of frames to sample pre and post step
            turn_idxs (np.array) = array containing indexes relating to when the participant was turning

    Returns:
        all_steps (np.array, number of steps x step maximum step length + buffer*2):  pressure for each step
        total_step_frame (np.array, size = (number of steps, maximum step length + buffer*2, D.shape[0], D.shape[1])) :
            pressure data for all steps

    """

    dim = D.shape
    participant = args.get('participant')
    buffer = args.get('buffer', 7)  # number of frames pre and post step initation and termiation
    plot = args.get('plot', False)
    if plot == True:
        output_path = args.get('output_path')

    D = np.nan_to_num(D, nan=0.)

    # create entire frame for each step
    # (number of steps, length of longest step + buffer (*2 for before and after step), data.dim[0], data.dim[1])
    total_step_frame = np.zeros((len(step_start[0]), int(np.max(index_differences)) + buffer*2, dim[0],dim[1]))  # save entire frame to an array

    all_steps = np.zeros((len(step_start[0]), int(np.max(index_differences)) + buffer*2))

    under_threshold = np.append(under_threshold[0], list(range(under_threshold[0][-1]+1, dim[2]+buffer*10)))

    exclude = 0
    plt.figure(figsize=(30,8*ceil(len(step_start[0])/5)))
    # loop through number of steps
    for i in range(len(step_start[0])):
        plt.subplot(ceil(len(step_start[0])/5), 5, i + 1)  # number of plots will be different for different participants
        plt.title('Step ' + str(i + 1))

        # list all indexes involved in the step
        step_indexes = list(range(under_threshold[step_start[0][i] - buffer], under_threshold[step_start[0][i] + buffer]))
        if any(np.isin(step_indexes, turn_idxs)) != True:

            if len(step_indexes) > total_step_frame.shape[1]: # if longer than the longest step

                continue

            else:

                # loop through frames involved in the step
                for j in range(len(step_indexes)):

                    if step_indexes[j] > D.shape[2]-1:
                        total_step_frame[i][j] = 0

                        all_steps[i][j] = 0
                    else:

                        # save frames to total_step_frame
                        total_step_frame[i][j] = D[:, :, step_indexes[j]]

                        # sum pressure for all frames involved in the step
                        all_steps[i][j] = np.sum(D[:, :, step_indexes[j]])

        else:
            continue

        plt.plot(all_steps[i])

    if plot == True:
        plt.subplots_adjust(hspace=0.7, wspace=0.5)
        plt.savefig(output_path + 'Single step pressures ' + str(participant) + ' ' + str(foot)+'.png')

    if plot == True:
        plt.figure()
        for i in range(all_steps.shape[0]):
            plt.plot(all_steps[i])
        plt.savefig(output_path + 'All steps ' + str(participant) + ' ' + str(foot)+'.png')

    av = np.mean(all_steps[:4], axis=0)
    peaks, _ = find_peaks(av, height=20000)
    if plot == True:
        plt.figure()
        plt.plot(av)
        plt.plot(peaks, av[peaks], "x")
        plt.savefig(output_path + 'Average pressure ' + str(participant) + ' ' + str(foot)+'.png')

    return  all_steps, total_step_frame

def normalize_step_length_given_longest(all_steps, total_step_frame, **args):
    """ Normalizes steps to one length through interpolation. Ignores the first and last step in the data.

    Args:
        all_steps (np.array, number of steps x step maximum step length + buffer*2):  pressure for each step
        total_step_frame (np.array, size = (number of steps, maximum step length + buffer*2, D.shape[0], D.shape[1])) :
            pressure data for all steps
        **args:
            longest (int): length to normalize step length to. default = 100 timepoints

    Returns:
        all_steps_new: normalised version of all_steps
        total_step_frame_new: normalised version of total_step_frame

    """

    longest = args.get('longest', 100)

    start_end = np.zeros((all_steps.shape[0], 2))

    # go through all steps, ignoring the first and last step
    for i in range(all_steps.shape[0] - 2):
        if np.sum(all_steps[i+1]) != 0: # skips out when there is no data for a step
            test = [np.where(all_steps[i + 1] > 0)[0][0] - 1, np.where(all_steps[i + 1] > 0)[0][-1] + 4]

            if test[0] < 0: #
                test += (0 - test[0])
            start_end[i,0], start_end[i,1] = test[0], test[1]
        else:
            continue

    all_steps_new = np.zeros((0, int(longest)))
    ref = list(range(int(longest)))


    for i in range(all_steps.shape[0] - 2):
        if int(start_end[i][1]) - int(start_end[i][0]) == 0.0:
            continue
        else:

            new = np.moveaxis(total_step_frame[i + 1][int(start_end[i][0]): int(start_end[i][1])], 0, 2)
            data = all_steps[i+1][int(start_end[i][0]): int(start_end[i][1])]

            arr1_interp = interpolate.interp1d(np.arange(len(data)), data)
            arr1_compress = arr1_interp(np.linspace(0, (len(data))-1, len(ref)))

            all_steps_new = np.vstack((all_steps_new, arr1_compress))


    # shape = number of steps x width of insole x length of insole x longest step
    total_step_frame_new = np.zeros((total_step_frame.shape[0] - 2, longest, total_step_frame.shape[2], total_step_frame.shape[3]))
    for i in range(total_step_frame.shape[0]-2):
        if int(start_end[i+1][1]) - int(start_end[i+1][0]) == 0.0:
            continue
        else:

            for j in range(total_step_frame.shape[2]):

                for k in range(total_step_frame.shape[3]):

                    data = total_step_frame[i+1,int(start_end[i+1][0]):int(start_end[i+1][1]),j,k]

                    arr1_interp = interpolate.interp1d(np.arange(len(data)), data)
                    arr1_compress = arr1_interp(np.linspace(0, (len(data)) - 1, len(ref)))

                    total_step_frame_new[i,:,j,k] = arr1_compress

    return all_steps_new, total_step_frame_new

def average_pressure_per_region(reshaped_data, regions, foot, participant, **args):
    """

    Args:
        reshaped_data (np.array()): 2D array of pressure data
        regions (list): list of sensor locations
        foot (str): foot (left or right)
        participant (str): Participant ID
        **args:
            plot (bool): if True, plot

    Returns:
        av_pressure_per_region (dict): mean pressure at each region
        contact_percentage_per_region (dict)
        total_pressure_per_region (dict): pressure at each region

    """

    plot = args.get('plot', False)
    if plot == True:
        output_path = args.get('output_path')

    # number of timepoints in data
    frames = reshaped_data.shape[1]

    # find region names
    region_names = fs.constants.foot_tags

    # create array to store total pressure per region
    total_pressure_per_region = dict.fromkeys(region_names, np.zeros(frames))

    reshaped_data[reshaped_data < 1.] = 0.0

    for region in total_pressure_per_region:
        region_indexes = np.argwhere(regions == region)
        total_pressure_per_region[region] = np.sum(reshaped_data[region_indexes], axis=0)


    av_pressure_per_region = dict.fromkeys(region_names, np.zeros(frames))


    for region in av_pressure_per_region:
        region_indexes = np.argwhere(regions == region)
        av_pressure_per_region[region] = np.mean(reshaped_data[region_indexes], axis=0)

    if plot == True:
        plt.figure(figsize=(15, 10))
        plt.suptitle('Average pressure per region')
        for i in range(len(av_pressure_per_region)):
            plt.subplot(3, 5, i + 1)
            plt.plot(av_pressure_per_region[region_names[i]][0])
            plt.title(region_names[i])
        plt.savefig(output_path + 'Average pressure per region ' + str(participant) + ' ' + str(foot) + '.png')



    contact_percentage_per_region = dict.fromkeys(region_names, np.zeros(frames))
    for i in range(len(contact_percentage_per_region)):
        contact_percentage_per_region[region_names[i]] = len(np.where(av_pressure_per_region[region_names[i]] > 0)[0]) / frames


    if plot == True:
        plt.figure()
        plt.bar(contact_percentage_per_region.keys(), contact_percentage_per_region.values())
        plt.savefig(output_path + 'Contact percentage per region' + participant + ' ' + foot + '.png')

    return av_pressure_per_region, contact_percentage_per_region, total_pressure_per_region

def expected_participant_mass_per_region(total_pressure_per_region, regions, reshaped_data, **args):
    """ Calculate force (in % body mass) experienced by each region of the foot

    Args:
        total_pressure_per_region (dict): pressure at each region
        reshaped_data (np.array()): 2D array of pressure data
        regions (list): list of sensor locations
        **args:
            sensor_area (float): area of each sensor (in m^2)

    Returns:
        expected_mass_per_region (dict): force (% body mass) experienced by each region
    """
    sensor_area = args.get('sensor_area', .0000258064)  # m^2

    expected_mass_per_region = {}


    reshaped_data[reshaped_data < 1.] = 0.0
    # convert pressure to N/m^2
    kpa_to_pa = reshaped_data * 1000

    # multiply by area to get force (N)
    times_by_sensor_area = kpa_to_pa * sensor_area

    # divide by gravity to get mass
    divide_by_gravity = times_by_sensor_area / 9.80665

    # loop through each region of the foot
    for region in fs.constants.foot_tags:

        # find sensors that are mapped onto the region in question
        indexes = np.where(regions == region)

        expected_mass_per_region[region] = np.nansum(divide_by_gravity[indexes], axis=0)


    return expected_mass_per_region

def expected_mass_coarse_regions(mass, expected_mass_per_region, **args):
    """ Calculate the force experienced (in % body mass) per each of the 4 coarse regions of the foot
        (toes, metatarsals, arch, heel)

    Args:
        mass (float): participant mass (kg)
        expected_mass_per_region (dict): force (% body mass) experienced by each region
        **args:

    Returns:
        toes (np.array()): array containing force experienced at the toes
        metatarsal (np.array()): array containing force experienced at the metatarsals
        arch (np.array()): array containing force experienced at the arch
        heel (np.array()): array containing force experienced at the heel

    """

    remove_less_than_zero = args.get('remove', True)

    # create empty array for each region to store the data in
    toes = np.zeros((0,expected_mass_per_region['T1'].shape[0]))
    metatarsal = np.zeros((0, expected_mass_per_region['T1'].shape[0]))
    arch = np.zeros((0, expected_mass_per_region['T1'].shape[0]))
    heel = np.zeros((0, expected_mass_per_region['T1'].shape[0]))

    # group regions together into the 4 coarse regions
    for region in expected_mass_per_region:
        if region[0] == 'T':
            toes = np.vstack((toes,expected_mass_per_region[region]))
        elif region[0] == 'M':
            metatarsal = np.vstack((metatarsal, expected_mass_per_region[region]))
        elif region[0] == 'A':
            arch = np.vstack((arch, expected_mass_per_region[region]))
        else:
            heel = np.vstack((heel, expected_mass_per_region[region]))


    # calculate force in % body mass
    toes = (np.nansum(toes, axis=0) / mass) * 100
    toes[toes < 0] = 0
    metatarsal = (np.nansum(metatarsal, axis=0) / mass) * 100
    metatarsal[metatarsal < 0] = 0
    arch = (np.nansum(arch, axis=0) / mass) * 100
    arch[arch < 0] = 0
    heel = (np.nansum(heel, axis=0) / mass) * 100
    heel[heel < 0] = 0

    if remove_less_than_zero == True:
        toes = toes[toes != 0]
        metatarsal = metatarsal[metatarsal != 0]
        arch = arch[arch != 0]
        heel = heel[heel != 0]


    return toes, metatarsal, arch, heel


def total_pressure_coarse_regions(pressure_per_region, **args):
    """ Calculate the total pressure experienced (in kPa) per each of the 4 coarse regions of the foot
        (toes, metatarsals, arch, heel)

    Args:
        pressure_per_region (dict): pressure (kPa) experienced by each region
        **args:

    Returns:
        toes (np.array()): array containing pressure experienced at the toes
        metatarsal (np.array()): array containing pressure experienced at the metatarsals
        arch (np.array()): array containing pressure experienced at the arch
        heel (np.array()): array containing pressure experienced at the heel

    """

    remove_less_than_zero = args.get('remove', True)


    # create empty array for each region to store the data in
    toes = np.zeros((0,pressure_per_region['T1'].shape[1]))
    metatarsal = np.zeros((0, pressure_per_region['T1'].shape[1]))
    arch = np.zeros((0, pressure_per_region['T1'].shape[1]))
    heel = np.zeros((0, pressure_per_region['T1'].shape[1]))


    for region in pressure_per_region:
        if region[0] == 'T':
            toes = np.vstack((toes,pressure_per_region[region]))
        elif region[0] == 'M':
            metatarsal = np.vstack((metatarsal, pressure_per_region[region]))
        elif region[0] == 'A':
            arch = np.vstack((arch, pressure_per_region[region]))
        else:
            heel = np.vstack((heel, pressure_per_region[region]))

    toes = np.nansum(toes, axis=0)
    metatarsal = np.nansum(metatarsal, axis=0)
    arch = np.nansum(arch, axis=0)
    heel = np.nansum(heel, axis=0)

    if remove_less_than_zero == True:
        toes = toes[toes != 0]
        metatarsal = metatarsal[metatarsal != 0]
        arch = arch[arch != 0]
        heel = heel[heel != 0]


    return toes, metatarsal, arch, heel


def average_pressure_coarse_regions(pressure_per_region, **args):
    """ Calculate the average pressure experienced (in kPa) per each of the 4 coarse regions of the foot
        (toes, metatarsals, arch, heel)

    Args:
        pressure_per_region (dict): pressure (kPa) experienced by each region
        **args:

    Returns:
        toes (np.array()): array containing pressure experienced at the toes
        metatarsal (np.array()): array containing pressure experienced at the metatarsals
        arch (np.array()): array containing pressure experienced at the arch
        heel (np.array()): array containing pressure experienced at the heel

    Args:
        pressure_per_region:
        **args:

    Returns:

    """
    remove_less_than_zero = args.get('remove', True)

    # create empty array for each region to store the data in
    toes = np.zeros((0,pressure_per_region['T1'].shape[1]))
    metatarsal = np.zeros((0, pressure_per_region['T1'].shape[1]))
    arch = np.zeros((0, pressure_per_region['T1'].shape[1]))
    heel = np.zeros((0, pressure_per_region['T1'].shape[1]))


    for region in pressure_per_region:
        if region[0] == 'T':
            toes = np.vstack((toes,pressure_per_region[region]))
        elif region[0] == 'M':
            metatarsal = np.vstack((metatarsal, pressure_per_region[region]))
        elif region[0] == 'A':
            arch = np.vstack((arch, pressure_per_region[region]))
        else:
            heel = np.vstack((heel, pressure_per_region[region]))

    toes = np.nanmean(toes, axis=0)
    metatarsal = np.nanmean(metatarsal, axis=0)
    arch = np.nanmean(arch, axis=0)
    heel = np.nanmean(heel, axis=0)

    if remove_less_than_zero == True:
        toes = toes[toes != 0]
        metatarsal = metatarsal[metatarsal != 0]
        arch = arch[arch != 0]
        heel = heel[heel != 0]


    return toes, metatarsal, arch, heel

def check4steps(D, **args):
    """ Checks for steps within the pressure data

    Args:
        D (np.array()): pressure matrix
        **args:
            threshold (int): total pressure threshold to ensure noise does not influence identification of when foot
                                is off the ground
            plot (bool): if True, plot

    Returns:
        under_threshold (np.array): indexes of frames where total pressure was below a specified threshold
                (assuming this is when the foot was off the floor)
        step_start: indexes of when each step begins
        index_differences: difference between two indexes. when larger than 1, it is assumed a step has taken place
        total_pressure:

    """

    threshold = args.get('threshold', 500)
    plot = args.get('plot', False)

    D = np.nan_to_num(D, nan=0.)

    total_pressure = np.zeros(D.shape[2])
    # calculate total pressure per frame
    for i in range(D.shape[2]):
        total_pressure[i] = np.sum(D[:, :, i])

    if plot == True:
        output_path = args.get('output_path')

        plt.figure()
        plt.plot(total_pressure)
        plt.savefig(output_path + 'total pressure over time.png')

    # break data down into steps
    # find indexes where total pressure is below a given threshold
    under_threshold = np.where(total_pressure < threshold)

    index_differences = []
    for i in range(len(under_threshold[0]) - 1):
        index_differences = np.append(index_differences, under_threshold[0][i + 1] - under_threshold[0][i])


    if len(index_differences) == 0:
        print('There are no steps in this data')
        step_start = np.zeros((0,0))
    else:
        # find where index jumps (i.e large between indexes is where a step happened)
        # find gaps between indexes
        # find indexes were difference is greater than 1
        step_start = np.where(index_differences > 1)
        print('There are ' + str(len(step_start[0])) + ' steps in this data')

    return under_threshold, step_start, index_differences, total_pressure

def compressed_pickle(title, data):
    """ Compress a pickle (.pk/.pkl) file to a .pbz2 file

    Args:
        title (str): filename
        data (var): varaible

    Returns:
        Saves variable as a .pbz2 file

    """
    with bz2.BZ2File(title + '.pbz2', 'wb') as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    """ Open a .pbz2 file

    Args:
        file (str): filename

    Returns:
        file contents

    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def left_right_both_mass(left_data, right_data, separation_indexes, actual_mass, participant, trial):
    """ Calculate the expected mass at the left foot, right foot and when on both feet - used for insole validation

    Args:
        left_data (np.array()): data for the left foot
        right_data (np.array()): data for the right foot
        separation_indexes (int): timepoints to split data into single foot stance and both feet stance (in project_insole_constants)
        actual_mass (float): participant mass (kg)
        participant (str): participant ID
        trial (str): trial ID

    Returns:
        left_only_mass (np.array()): expected mass when on single foot stance on left foot
        right_only_mass (np.array()): expected mass when on single foot stance on right foot
        both_mass (np.array()): expected mass when on both foot stance

    """
    # get period of time where only the left foot is on the ground
    left_only_mass = (expected_participant_weight(left_data[:, :,
                                                  separation_indexes[trial][participant]['left'][0]:
                                                  separation_indexes[trial][participant]['left'][
                                                      1]]) / actual_mass) * 100
    # get period of time where only the right foot is on the ground√
    right_only_mass = (expected_participant_weight(right_data[:, :,
                                                   separation_indexes[trial][participant]['right'][0]:
                                                   separation_indexes[trial][participant]['right'][
                                                       1]]) / actual_mass) * 100

    # get period of time where both feet are on the ground
    left_both_mass = (expected_participant_weight(left_data[:, :,
                                                  separation_indexes[trial][participant]['both'][0]:
                                                  separation_indexes[trial][participant]['both'][
                                                      1]]) / actual_mass) * 100
    right_both_mass = (expected_participant_weight(right_data[:, :,
                                                   separation_indexes[trial][participant]['both'][0]:
                                                   separation_indexes[trial][participant]['both'][
                                                       1]]) / actual_mass) * 100
    both_mass = left_both_mass + right_both_mass  # add the left and right foot together

    return left_only_mass, right_only_mass, both_mass

def heel_tiptoe_mass(left_data, right_data, separation_indexes, actual_mass, participant, trial):
    """ Calculate the expected mass at the left foot, right foot and when on both feet - used for insole validation

        Args:
            left_data (np.array()): data for the left foot
            right_data (np.array()): data for the right foot
            separation_indexes (int): timepoints to split data into single heel stance and tiptoe stance (in project_insole_constants)
            actual_mass (float): participant mass (kg)
            participant (str): participant ID
            trial (str): trial ID

        Returns:
            heel_mass (np.array()): expected mass when in heel stance
            tiptoe_mass (np.array()): expected mass when in tiptoe stance

    """


    # get period of time where participant is on tiptoes for the left foot
    left_tiptoe_mass = (expected_participant_weight(left_data[:, :,
                                                    separation_indexes[trial][participant]['tiptoe'][0]:
                                                    separation_indexes[trial][participant]['tiptoe'][
                                                        1]]) / actual_mass) * 100
    # get period of time where is on tiptoes for the right foot
    right_tiptoe_mass = (expected_participant_weight(right_data[:, :,
                                                     separation_indexes[trial][participant]['tiptoe'][0]:
                                                     separation_indexes[trial][participant]['tiptoe'][
                                                         1]]) / actual_mass) * 100
    tiptoe_mass = left_tiptoe_mass + right_tiptoe_mass  # sum the left and right foot

    # get period of time where participant is on heels for the left foot
    left_heel_mass = (expected_participant_weight(left_data[:, :,
                                                  separation_indexes[trial][participant]['heel'][0]:
                                                  separation_indexes[trial][participant]['heel'][
                                                      1]]) / actual_mass) * 100
    # get period of time where participant is on heels for the right foot
    right_heel_mass = (expected_participant_weight(right_data[:, :,
                                                   separation_indexes[trial][participant]['heel'][0]:
                                                   separation_indexes[trial][participant]['heel'][
                                                       1]]) / actual_mass) * 100
    heel_mass = left_heel_mass + right_heel_mass  # sum up data from the left and right foot

    return heel_mass, tiptoe_mass

def read_left_foot_accel_csv(filename, **args):
    """ Identify the first stomp in the data using the acceleration data from the left foot IMU signal (in the z axis)

    Args:
        filename (str): filepath to acceleration file from the IMU
        **args:
            plot (bool): if True, plot

    Returns:
        peaks[0] (int): timepoint at which the first peak happens - identifying the stomp

    """



    plot = args.get('plot', False)
    if plot == True:
        output_path = args.get('output_path')

    left_foot_accel_data = np.asarray(pd.read_csv(filename)).T

    # calculate resultant
    x_resultant = left_foot_accel_data[0]**2
    y_resultant = left_foot_accel_data[1]**2
    z_resultant = left_foot_accel_data[2]**2

    total = x_resultant+y_resultant+z_resultant

    resultant = np.zeros(len(total))
    for i in range(len(total)):
        resultant[i] = sqrt(total[i])

    # identify timepoints during which the acceleration is greatest
    peaks, _ = find_peaks(resultant, height=30)

    if plot == True:
        plt.figure()
        plt.suptitle('Left foot acceleration data')
        plt.subplot(1,3,1)
        plt.title('x')
        plt.plot(left_foot_accel_data[0], c='b')
        plt.axvline(x=peaks[0], c='r', lw=1)

        plt.subplot(1,3,2)
        plt.title('y')
        plt.plot(left_foot_accel_data[1],c='b')
        plt.axvline(x=peaks[0], c='r', lw=1)

        plt.subplot(1,3,3)
        plt.title('z')
        plt.plot(left_foot_accel_data[2],c='b')
        plt.axvline(x=peaks[0], c='r', lw=1)
        #plt.plot(peaks, np.abs(left_foot_accel_data[2])[peaks], "x", c='orange')
        plt.show()
        plt.savefig(output_path + 'Identifying IMU Stomp.png')

    return peaks[0]

def localise_CoP(loc, CoP_location):
    """ Identifies the region at which the CoP is located.
        Sometimes the CoP can land in the small gaps between regions due to the design on the foot within FootSim.
        If this happens, the nearest region is identified, first along the x axis. If this doesn't lead to a region,
        then the nearest region is found along the y axis

    Args:
        loc (np.array()): coordinates of the CoP
        CoP_location (list): list to store locations in CoP

    Returns:
        CoP_location (list): list of CoP locations
    """
    # calculate centre of pressure location
    for location in loc:
        if np.isnan(location[0]) or np.isnan(location[1]):
            CoP_location.append('NA')
        else:
            foot_region = fs.foot_surface.locate(location)[0][0]
            if foot_region == '':
                foot_region = fs.foot_surface.locate([location[0] - 5, location[1]])[0][0]

                if foot_region == '':
                    foot_region = fs.foot_surface.locate([location[0] + 5, location[1]])[0][
                        0]

                    if foot_region == '':

                        foot_region = \
                            fs.foot_surface.locate([location[0] - 10, location[1]])[0][0]

                        if foot_region == '':
                            foot_region = \
                                fs.foot_surface.locate([location[0], location[1] - 5])[0][0]

                            if foot_region == '':

                                CoP_location.append('NA')

                            else:
                                CoP_location.append(foot_region[0])

                        else:
                            CoP_location.append(foot_region[0])
                    else:
                        CoP_location.append(foot_region[0])

                else:
                    CoP_location.append(foot_region[0])
            else:
                CoP_location.append(foot_region[0])

    return CoP_location


def scale_matrix(target, original):
    """ Upscales matrix of a smaller foot to the size of a larger foot through interpolation

    Args:
        target (np.array()): target matrix i.e. matrix of largest foot
        original (np.array()): matrix to be upscaled

    Returns:
        scaled (np.array()): upscaled version of the original matrix
    """

    scaled = np.zeros((target.shape[0], target.shape[1], original.shape[2]))

    for i in range(original.shape[2]):
        scaled[:, :, i] = resize(original[:, :, i], (target.shape[0], target.shape[1]))


    return scaled


def centre_of_contact(D, **args):
    """ Find the centre of contact trace throughout the data

    Args:
        D: Pressure insole data

    Returns:
        loc: co-ordinates of centre of contact coordinates in foot space ([y, x])
        y: array containing y coordinates for centre of contact in pixel space
        x: array containing x coordinates for centre of contact in pixel space

    """
    threshold = args.get('threshold', 500)

    D = np.nan_to_num(D, nan=0)
    contact = deepcopy(D)

    # active sensors set to 1 to provide equal weight
    # inactive sensors set to 0
    contact[contact < 1] = 0
    contact[contact > 1] = 1

    dim = D.shape  # Dimensions
    # Get foot boundary coordinates
    x_outline, y_outline = get_foot_outline()

    cmin = [min(x_outline), min(y_outline)]
    cmax = [max(x_outline), max(y_outline)]

    c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    c1 = np.linspace(cmin[0], cmax[0], dim[1])

    #dim = D.shape  # Dimensions
    #cmin = np.min(fs.foot_surface.bbox_min, axis=0)  # Calculates bounding box for arbitrary boundary. # y
    #cmax = np.max(fs.foot_surface.bbox_max, axis=0)

    #c0 = np.flip(np.linspace(cmin[1], cmax[1], dim[0]))
    #c1 = np.linspace(cmin[0], cmax[0], dim[1])

    loc = np.zeros((0, 2))
    x = list()
    y = list()


    for i in range(dim[2]):

        if np.sum(D[:, :, i]) < threshold:
            frame_COC = [np.nan, np.nan]

            loc = np.vstack((loc, frame_COC))

            y = np.append(x, frame_COC[1])
            x = np.append(y, frame_COC[0])

        else:

            # find coordinates of centre of mass
            frame_COC = ndimage.center_of_mass(contact[:, :, i])
            frame_COC = np.nan_to_num(frame_COC, nan=0.)

            diff_one_x = c1[math.ceil(frame_COC[1])] - c1[math.floor(frame_COC[1])]
            diff_two_x = frame_COC[1] - int(frame_COC[1])
            add_x = 1 / diff_two_x
            ans_x = c1[math.floor(frame_COC[1])] + (diff_one_x / add_x)

            diff_one_y = c0[math.ceil(frame_COC[0])] - c0[math.floor(frame_COC[0])]
            diff_two_y = frame_COC[0] - int(frame_COC[0])
            add_y = 1 / diff_two_y
            ans_y = c0[math.floor(frame_COC[0])] + (diff_one_y / add_y)

            x = ans_x
            y = ans_y

            # transform coordinates into foot space
            loca = np.array([ans_x, ans_y])
            #loca = fs.foot_surface.pixel2hand(np.array([ans_x, ans_y]))

            #print(loca)

            loc = np.vstack((loc, loca))

            y = np.append(x, frame_COC[1])
            x = np.append(y, frame_COC[0])

    return loc, y, x


def foot_on_foot_off(total_pressure, **args):
    """ Calculate when the foot is on the ground and return binary array (1 = on the ground, 0 = off the ground)

    :param total_pressure:
    :return:
    """

    threshold = args.get('threshold', 500)

    on_off = np.zeros(len(total_pressure))
    for i in range(len(total_pressure)):
        if total_pressure[i] > threshold:
            on_off[i] = 1

    return on_off


def CoP_inside_outside_contact_area(reshaped_data, s, CoP_coords, **args):

    overall_threshold = args.get('overall_threshold', 500)
    sensor_threshold = args.get('sensor_threshold', 1)

    in_out = np.zeros(reshaped_data.shape[1])
    percentile = np.zeros(reshaped_data.shape[1])

    tree = spatial.KDTree(s.location)

    for i in range(reshaped_data.shape[1]):

        if np.sum(reshaped_data[:,i]) < overall_threshold:
            in_out[i] = np.nan
        else:

            CoP = CoP_coords[i]

            nearest = tree.query(CoP, 4)

            if np.any(np.isinf(nearest[0])) == True:
                percentile[i] = 0
                in_out[i] = 0

            elif np.all(reshaped_data[:,i][nearest[1]] > sensor_threshold) == True:
            #if reshaped_data[nearest[1]][i] > sensor_threshold:
                percentile[i] = stats.percentileofscore(reshaped_data[:,i], np.mean(reshaped_data[:,i][nearest[1]]))
                in_out[i] = 1
            else:
                percentile[i] = stats.percentileofscore(reshaped_data[:, i], np.mean(reshaped_data[:, i][nearest[1]]))
                in_out[i] = 0

    return in_out, percentile

