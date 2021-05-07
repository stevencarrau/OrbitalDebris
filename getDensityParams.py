
def getDensityParams(altitude):
    # %
    # % function [rho0,h0,H] = getDensityParams( altitude )
    # % ---------------------------------------------------------------------
    # %
    # % Description:
    # %
    # %  Get the density model parameters for the 1976 Standard Atmosphere
    # %  exponential model
    # %
    # % Inputs:
    # %
    # %  altitude - Altitude above the Earth's surface in kilometers
    # %
    # % Outputs:
    # %
    # %  rho0 - Nominal density in kg/m^3
    # %  h0   - Base altitude (not radius!) in km
    # %  H    - Scale height in km
    # %
    # % Assumptions/References:
    # %
    # %  Vallado and McClain, Third Edition, P. 564
    # %
    # % Dependencies:
    # %
    # %  None
    # %
    # % Modification History:
    # %
    # %   18jan17     Brandon A. Jones      original version
    # %
    # % ---------------------------------------------------------------------
    # % Copyright University of Texas at Austin, 2017
    # %


    if altitude > 1000:
        rho0 = 3.019e-15
        h0 = 1000
        H = 268
    elif altitude > 900:
        rho0 = 5.245e-15
        h0 = 900
        H = 181.05
    elif altitude > 800:
        rho0 = 1.170e-14
        h0 = 800
        H = 124.64
    elif altitude > 700:
        rho0 = 3.614e-14
        h0 = 700
        H = 88.667
    elif altitude > 600:
        rho0 = 1.454e-13
        h0 = 600
        H = 71.835
    elif altitude > 500:
        rho0 = 6.967e-13
        h0 = 500
        H = 63.822
    elif altitude > 450:
        rho0 = 1.585e-12
        h0 = 450
        H = 60.828
    elif altitude > 400:
        rho0 = 3.725e-12
        h0 = 400
        H = 58.515
    elif altitude > 350:
        rho0 = 9.518e-12
        h0 = 350
        H = 53.298
    elif altitude > 300:
        rho0 = 2.418e-11
        h0 = 300
        H = 53.628
    elif altitude > 250:
        rho0 = 7.248e-11
        h0 = 250
        H = 45.546
    elif altitude > 200:
        rho0 = 2.789e-10
        h0 = 200
        H = 37.105
    elif altitude > 180:
        rho0 = 5.464e-10
        h0 = 180
        H = 29.740
    elif altitude > 150:
        rho0 = 2.070e-9
        h0 = 150
        H = 22.523
    elif altitude > 140:
        rho0 = 3.845e-9
        h0 = 140
        H = 16.149
    elif altitude > 130:
        rho0 = 8.484e-9
        h0 = 130
        H = 12.636
    elif altitude > 120:
        rho0 = 2.438e-8
        h0 = 120
        H = 9.473
    elif altitude > 110:
        rho0 = 9.661e-8
        h0 = 110
        H = 7.263
    elif altitude > 100:
        rho0 = 5.297e-7
        h0 = 100
        H = 5.877
    elif altitude > 90:
        rho0 = 3.396e-6
        h0 = 90
        H = 5.382
    elif altitude > 80:
        rho0 = 1.905e-5
        h0 = 80
        H = 5.799
    elif altitude > 70:
        rho0 = 8.770e-5
        h0 = 70
        H = 6.549
    elif altitude > 60:
        rho0 = 3.206e-4
        h0 = 60
        H = 7.714
    elif altitude > 50:
        rho0 = 1.057e-3
        h0 = 50
        H = 8.382
    elif altitude > 40:
        rho0 = 3.972e-3
        h0 = 40
        H = 7.554
    elif altitude > 30:
        rho0 = 1.774e-2
        h0 = 30
        H = 6.682
    elif altitude > 25:
        rho0 = 3.899e-2
        h0 = 25
        H = 6.349
    else:
        rho0 = 1.225
        h0 = 0
        H = 7.249

    return rho0,h0,H