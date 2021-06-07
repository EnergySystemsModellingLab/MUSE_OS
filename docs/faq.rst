FAQs
====

I get a demand matching error
-----------------------------

If you get an error, similar to the following:

.. code-block:: python

    muse/src/muse/investments.py", line 328, in scipy_match_demand
        raise LinearProblemError("LP system could not be solved", res)
    muse.investments.LinearProblemError: ('LP system could not be solved',      con: array([], dtype=float64)
        fun: 15053.021584161033
    message: 'The algorithm terminated successfully and determined that the problem is infeasible.'
        nit: 11
    slack: array([ 0.12188998,  2.27624361,  1.72261721,  0.65248747, -0.22047042,
            0.62430284,  5.80377518,  1.96572729,  1.5198397 ,  0.61823037,
        -0.21871122,  0.59457721,  2.49343834,  1.10418509,  0.84691291,
            0.3260397 , -0.2204754 ,  0.31897793,  4.36788165,  4.9152109 ,
            6.91218128, 23.93103992,  8.76641572,  7.35194341,  3.24577114,
        -0.26032861,  2.8121707 ])
    status: 2
    success: False
        x: array([32.03211835, 35.0847891 , 43.08781872, 11.8901544 ,  3.72977858,
            3.08220055,  1.74992141,  1.42167486,  1.77810604,  7.35302074,
            4.61267067,  3.74287866,  2.01312881,  1.53439081,  2.03678197,
            4.68786478,  2.48646647,  2.02560833,  1.11022092,  0.93860572,
            1.11728269]))


this is because the optimisation algorithm can not find a solution to match supply with demand. This is often because the constraints placed on the technologies do not allow for high enough growth to meet a growing demand.

A solution to this is to increase the limits of technologies in the relevant `Technodata.csv`. For example, by increasing the `MaxCapacityAddition`, `MaxCapacityGrowth` and/or `TotalCapacityLimit` variables for the respective technologies.




What units should I be using within MUSE?
-----------------------------------------

The units within MUSE should be consistent. Therefore it is up to you which units you use. You could use, like the examples, petajoules (PJ), however, the units used must be the same across each of the sectors, and each of the input files. MUSE does not make any unit conversion internally.


How do I activate my conda environment?
---------------------------------------

To activate your conda environment, run the following command:

.. code-block:: console

    conda activate <name-of-environment>

For example, if you want to activate your conda environment called muse run:

.. code-block:: console

    conda activate muse


How do I know my conda environment is activated?
------------------------------------------------

Your conda environment is activated if you see something similar to the following in your Anaconda Powershell Prompt or command line:

.. code-block:: console

    (muse) PS C:/Users/<my-username>


I get a "Cannot find command 'git' - do you have 'git' installed in your PATH?" error
---------------------------------------------------------------------------------------

This is because you do not have git installed in your conda environment. To resolve this, run:

.. code-block:: console

    conda install git

When I input my GitHub password into Anaconda Powershell Prompt to download MUSE, I don't see any input
-------------------------------------------------------------------------------------------------------

This is normal behaviour. It is done to stop people watching as you type your password over your shoulder. Just continue typing in your password as you would on a website.
