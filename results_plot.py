import math
import numpy as np
import matplotlib.pyplot as plt
import attack_analysis_data

x_1 = np.arange(0.10, 3.10, 0.10)
x_2 = np.arange(1.0/30, 31.0/30, 1.0/30) 
rmse_error = attack_analysis_data.rmse_error


plt.plot(x_1, rmse_error[0], x_1, rmse_error[1], x_1, rmse_error[2], x_1, rmse_error[3], x_1, rmse_error[4])
plt.axis([0.10, 3.2, 0, 0.5])
plt.xlabel('Atttack strength')
plt.ylabel('Error')
plt.legend(('16.6','33.3','50','66.6','83.3'), loc="best")
plt.show()

plt.plot(x_1, rmse_error[5], x_1, rmse_error[6], x_1, rmse_error[7], x_1, rmse_error[8], x_1, rmse_error[9])
plt.axis([0.10, 3.2, 0, 0.5])
plt.xlabel('Atttack strength')
plt.ylabel('Error')
plt.gca().legend(('0.10','0.20','0.30','0.40','0.50'), loc="best")
plt.show()

plt.plot(x_1, rmse_error[10], x_1, rmse_error[11], x_1, rmse_error[12], x_1, rmse_error[13], x_1, rmse_error[14])
plt.axis([0.10, 3.2, 0, 0.5])
plt.xlabel('Atttack strength')
plt.ylabel('Error')
plt.gca().legend(('0.10','0.20','0.30','0.40','0.50'), loc="best")
plt.show()

plt.plot(x_2, rmse_error[15], x_2, rmse_error[16], x_2, rmse_error[17], x_2, rmse_error[18], x_2, rmse_error[19])
plt.axis([1.0/30, 1.02, 0, 0.5])
plt.xlabel('Percentage of sensors under attack')
plt.ylabel('Error')
plt.gca().legend(('0.10','0.20','0.30','0.40','0.50'), loc="best")
plt.show()

plt.plot(x_2, rmse_error[20], x_2, rmse_error[21], x_2, rmse_error[22], x_2, rmse_error[23], x_2, rmse_error[24])
plt.axis([1.0/30, 1.02, 0, 0.5])
plt.xlabel('Percentage of sensors under attack')
plt.ylabel('Error')
plt.gca().legend(('0.10','0.20','0.30','0.40','0.50'), loc="best")
plt.show()

plt.plot(x_2, rmse_error[25], x_2, rmse_error[26], x_2, rmse_error[27], x_2, rmse_error[28], x_2, rmse_error[29])
plt.axis([1.0/30, 1.02, 0, 0.5])
plt.xlabel('Percentage of sensors under attack')
plt.ylabel('Error')
plt.gca().legend(('0.10','0.20','0.30','0.40','0.50'), loc="best")
plt.show()

#plt.plot(x_1, rmse_error[0], 'bo-', x_1, rmse_error[1], 'g*-', x_1, rmse_error[2], 'rs-',  x_1, rmse_error[3], 'cp-',  x_1, rmse_error[4], 'm^-')
