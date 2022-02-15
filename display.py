import matplotlib.pyplot as plt
import numpy as np
import formulas as f
import curvefitting as cf


class Histogram:
    def basic(list, bins,
              title=None, x_label=None, y_label=None):
        plt.hist(list, bins, range=[0, 7])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show(block=True)

    def lines(list, bins, low_cut, high_cut,
              title=None, x_label=None, y_label=None):
        n, bins, patches = plt.hist(list, bins, range=[0, 7])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(axis='x')
        plt.vlines(low_cut, 0, n.max(), color='red')
        plt.vlines(high_cut, 0, n.max(), color='red')
        plt.tight_layout()
        plt.show(block=True)


class Scatter:
    def scatter(df,
                title=None, x_label=None, y_label=None):
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        plt.scatter(x, y, s=2, alpha=0.5, cmap='bone')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


class ExpDecay:
    def onecomp(movie, df, equation, tau1, cov1, r2, kinetic,
                title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        val1 = np.round(tau1*movie.framestep_size, 3)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=0.5,
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '\n--- One Comp Exp Decay ---')
        plt.plot(x_data, equation(x_data, tau1),
                 'r--', label=f'{kinetic.name}: {val1} {kinetic.unit}\n'
                 + f' Cov: {var1}\n'
                 + f'R\u00b2: {r_val}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def twocomp(movie, df, a, tau1, cov1, tau2, cov2, r2, kinetic,
                title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        a1 = round(a, 3)*100
        a2 = round(100-a1, 3)
        val1 = round(tau1*movie.framestep_size, 3)
        val2 = round(tau2*movie.framestep_size, 3)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        var2 = np.format_float_scientific(cov2, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=0.5,
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(movie.fig_title + '\n --- Two Comp Exp Decay ---')
        plt.plot(x_data, cf.TwoCompExpDecay.equation(
            x_data, a, tau1, tau2), 'r--',
            label=f'Maj Frac: {a1}\n'
            + f' {kinetic.name}: {val1} {kinetic.unit}\n'
            + f' Cov {var1}\n'
            + f'Min Frac: {a2}\n'
            + f' {kinetic.name}: {val2} {kinetic.unit}\n'
            + f' Cov: {var2}\n'
            + f'R\u00b2: {r_val}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def explin(movie, df, tau1, cov1, b, cov2, r2, kinetic,
               title=None, x_label=None, y_label=None, legend=None):
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        val1 = round(tau1*movie.framestep_size, 4)
        val2 = np.format_float_scientific(b*movie.framestep_size,
                                          precision=1, exp_digits=2)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        var2 = np.format_float_scientific(cov2, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=0.5,
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '\n --- Exp and Lin Decay ---')
        plt.plot(x_data, cf.ExpLinDecay.equation(
            x_data, tau1, b), 'r--',
            label=f'Exp {kinetic.name}: {val1} {kinetic.unit}\n'
            + f' Exp Cov:{var1}\n'
            + f'Lin {kinetic.name}: {val2} frames\n'
            + f' Lin Cov: {var2}\n'
            + f'R\u00b2: {r_val}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


# class Rayleigh:
#     def twopart(self, df, a, a_cov, b, b_cov,
#                        sig1, sig1_cov, sig2, sig2_cov, r2,
#                        title=None, x_label=None, y_label=None, legend=None):
#         x_data = df.iloc[:, 0].values
#         y_data = df.iloc[:, 1].values
#         raypop1, raypop2 = a/(a+b), b/(a+b)
#         raydiffcoeff1 = round(1e8*sig1**2/(2*self.framestep_size), 4)
#         raydiffcoeff2 = round(1e8*sig2**2/(2*self.framestep_size), 4)
#         vara = np.format_float_scientific(a_cov, precision=1, exp_digits=2)
#         varb = np.format_float_scientific(b_cov, precision=1, exp_digits=2)
#         varsig1 = np.format_float_scientific(
#             sig1_cov, precision=1, exp_digits=2)
#         varsig2 = np.format_float_scientific(
#             sig2_cov, precision=1, exp_digits=2)
#         r_val = round(r2, 6)
#         plt.scatter(x_data, y_data, s=2, alpha=0.5,
#                     label=f'Data: n = {Formulas.traj_count(self.df)}\n\
#                        R2: {r_val}')
#         plt.xlabel(x_label)
#         plt.ylabel(y_label)
#         plt.title(title)
#         plt.plot(x_data, Fits.twopartRayeigh(
#             x_data, a, b, sig1, sig2), 'r--',
#             label=f'Maj Fraction: {raypop1} +/- {vara}\n \
#             Diff Coeff: {raydiffcoeff1} +/- {varsig1} um^2/sec\n\
#             Min Frac: {raypop2} +/- {varb}\n \
#             Diff Coeff: {raydiffcoeff2} +/- {varsig2} um^2/sec')
#         plt.legend(loc='upper right')
#         plt.show()
