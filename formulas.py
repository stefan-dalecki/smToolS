import numpy as np


class Calc:
    """
    Basic formulas to speed up calculations

    None of the math is particularly difficult but instead,
    offered are commonly used shortcut functions

    """

    def traj_count(df):
        """
        Count trajectories in a dataframe

        Quickly count the number of unique trajectories in a movie

        Args:
            df (df): dataframe containing 'Trajectory' column

        Returns:
            traj_count (int): number of trajectories in movie

        Raises:
            none

        """

        traj_count = df['Trajectory'].nunique()
        return traj_count

    def distance(x1, y1, x2, y2):
        """
        Compute distance between two points

        finds x and y values in dataframe, calculates distance between points

        Args:
            x1, y1, x2, y2 (float): coordinates

        Returns:
            traj_count (int): number of trajectories in movie

        Raises:
            none

        """

        distance = (((x2-x1)**2)+((y2-y1)**2))**(1/2)
        return distance

    def e_estimation(df):
        """
        Estimate likely time constant

        Find corresponding frame value of 1/e

        Args:
            df (df): bound state lifetime dataframe

        Returns:
            estimation (float): time constant estimation

        Raises:
            none

        """

        estimation = df.iloc[(df.iloc[:, 1]-(1/np.exp(1))).abs().argsort()[:1]]
        estimation = estimation.iloc[:, 0].values[0]
        return estimation

    def f_modeltest():
        """
        F-Test to Compare Two Models

        test whether the addition of parameters improves the model fit

        Args:
            sumsqr1 (float): residual sum of squares for simpler model
            sumsqr2 (float): residual sum of squares for complex model
            degfr1 (float): degrees of freedom for simpler model
            degfre2 (float): degrees of freedom for complex model

        Returns:
            f-statistic value

        Raises:
            error: degrees of freedom in model 2 are less than model 1

        """
        # try:
        #     f = ((sumsqr1 - sumsqr2) / (degfr1 - degfr2)) / \
        #         (sumsqr2 / degfr2)
        #     return f
        # except degfr1 > degfr2:
        #     print('Models are out of order.')
        pass

    def dof(lam=532, incidence=69, n2=1.33, n1=1.515):
        """
        smTIRF depth of field

        Calculate the 1/e light intensity point of the smTIRF setup

        Args:
            lam (int): laser wavenlength in nanometers
            incidence (float): laser incidence angle between 65.22 and 72.54
            n1 (float): sample refractive index, default is water
            n2 (float): coverslip refractive index, default is glass

        Returns:
            d (float): TIRF laser depth of field (nm)

        Raises:
            none

        """

        theta = incidence * (180 / np.pi)
        d = (lam / (4*np.pi)) * (n1**2 * (np.sin(theta))**2 - n2**2)**(-1/2)
        return d

    def lumi(I0, d, z):
        """
        Light intensity at 'z' distance

        smTIRF microscope light intensity gradient

        Args:
            I0 (float): initial light intensity
            d (float): TIRf laser depth of field (nm)
            z (float): distance from coverslip (nm)

        Returns:
            Iz (float): intensity at 'z' distance

        Raises:
            none

        """

        Iz = I0**(-z / d)
        return Iz


class Find:
    """
    Find sub data in large data

    example: Extract biologic features from file name

    """

    def identifiers(en_string, separator, val_name, IDs, nf='not found'):
        """
        Identify metrics from file name

        Find details like protein type, gasket, replicate, and so on

        Args:
            en_string (str): entire string
            separator (str): separator of interest in entire string
            val_name (str): dictionary key
            IDs (list): search strings for dictionary value
            nf (str): answer if no value is found

        Returns:
            output (dict): val_name (key) and value pair
                           two key-value pairs if multiple proteins

        Raises:
            Exception: keeps key (val_name) with NaN value

        """

        try:
            output = {}
            san_string = [i for i in en_string.lower().split(separator)]
            for sub_i, sub_str in enumerate(san_string):
                for ID in IDs:
                    exist = [i.find(ID) for i in san_string]
                    for exist_i, exist_val in enumerate(exist):
                        if exist_val != -1:
                            if '-' in san_string[exist_i]:
                                if '+' in san_string[exist_i]:
                                    proteins = san_string[exist_i].split('+')
                                    for molecule in proteins:
                                        conc, protein = molecule.split('-')
                                        conc = conc[:-1] + conc[-1].upper()
                                        protein = protein.upper()
                                        units = conc[2:]
                                        output[f'{val_name} ({units})'] = \
                                            f'{protein} ({conc[:-2]})'
                                else:
                                    conc, protein = san_string[exist_i].split(
                                        '-')
                                    conc = conc[:-1] + conc[-1].upper()
                                    units = conc[2:]
                                    protein = protein.upper()
                                    output[f'{val_name} ({units})'] = \
                                        f'{protein} ({conc[:-2]})'
                            else:
                                output = {
                                    val_name: str(san_string[exist_i].strip(ID))}
                                break
            if output:
                return output
            else:
                return {val_name: nf}
        except Exception:
            return {val_name: nf}


class Form:
    """
    Find sub data in large data

    example: Extract biologic features from file name

    """

    def catdict(*dicts):
        """
        Concatenate dictionaries

        Split dictionary key-value pairs into pairs for figure titles

        Args:
            *dicts (dict): key-value data

        Returns:
            catdict (dict): concatenated dictionary

        Raises:
            none

        """

        inter = []
        catdict = ' '
        for dict in dicts:
            for val in dict.items():
                st = ': '.join(val)
                inter.append(st)
        for i, str in enumerate(inter):
            if (i+1) % 2 == 0:
                catdict += ' // ' + str + '\n'
            else:
                catdict += str
        return catdict

    def reorder(df, col_name, loc):
        """
        Move df columns

        Change the location of columns by column name and locatin index

        Args:
            col_name (str): column name
            loc (int): index location of new column location

        Returns:
            df (df): rearranged dataframe

        Raises:
            AssertionError: column name not in df

        """

        assert col_name in df.columns
        col = df[col_name].values
        df = df.drop(columns = [col_name])
        df.insert(loc, col_name, col)
        return df

    def userinput(str, list):
        """
        User input interface

        Allow user to select option from list

        Args:
            str (str): option category
            list (list): list of possible options

        Returns:
            ans (str): option selection

        Raises:
            none

        """

        print(f'\nAvailable options\n{list}', end = '\n'*2)
        while True:
            ans = input(f'From options above, select your {str}: ')
            if ans not in list:
                print('Not an available option\n')
            else:
                break
        return ans

    def inputbool(str):
        """
        User input boolean

        Input answer is converted into boolean

        Args:
            str (str): user question

        Returns:
            ans (bool): True/False response to str user question

        Raises:
            none

        """

        while True:
            ans = input(str)
            if ans == 'y':
                ans = True
                break
            elif ans == 'n':
                ans = False
                break
            else:
                print('Not an available option\n')
        return ans

    def batching(task, script, export, batch_size, filelist):
        """
        Batch process movies

        Analyze multiple movies simultaneously

        Args:
            task: processing function
            script: script class object
            export: export class object
            num_processes (int): batch size
            filelist (list): list of files to analyze

        Returns:
            concatenated dataframe

        Raises:
            Assert Error: if the number of processes is <= 1
            Assert Error: if filetype is not 'csv'

        """
        
        from multiprocessing import Pool


        assert batch_size > 1
        assert script.filetype == 'csv'
        dfs = []
        while filelist:
            if batch_size >= len(filelist):
                batch_size = len(filelist)
            with Pool(batch_size) as pool:
                scripts = [script for i in range(batch_size)]
                dfs += list(pool.starmap(task, zip(filelist, scripts)))
                pool.close()
                pool.join()
            filelist = filelist[batch_size:]
        return pd.concat(dfs, ignore_index=True)
