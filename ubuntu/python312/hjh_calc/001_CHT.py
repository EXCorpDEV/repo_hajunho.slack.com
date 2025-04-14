import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Data range
x = np.linspace(0, 100, 1000)  # 0-100 TB

# Cost functions
cloud_cost = 1 + 0.25 * x       # Cloud: Initial 1M + 0.25M/TB
hybrid_cost = 5 + 0.15 * x      # Hybrid: Initial 5M + 0.15M/TB
onprem_cost = 10 + 0.1 * x      # On-premises: Initial 10M + 0.1M/TB

# Intersection points calculation
# Cloud = Hybrid intersection
# 1 + 0.25x = 5 + 0.15x
# 0.1x = 4
# x = 40
intersect1_x = 40
intersect1_y = 1 + 0.25 * intersect1_x

# Hybrid = On-premises intersection
# 5 + 0.15x = 10 + 0.1x
# 0.05x = 5
# x = 100
intersect2_x = 100
intersect2_y = 5 + 0.15 * intersect2_x

# Convex Hull Trick algorithm
def min_cost(x_val):
    cloud = 1 + 0.25 * x_val
    hybrid = 5 + 0.15 * x_val
    onprem = 10 + 0.1 * x_val
    return min(cloud, hybrid, onprem)

min_costs = [min_cost(x_val) for x_val in x]

# Matplotlib setup
plt.figure(figsize=(12, 8))
plt.style.use('ggplot')

# Draw cost function lines
plt.plot(x, cloud_cost, label='Cloud: 1M + 0.25M/TB', linewidth=2, color='#3498DB')
plt.plot(x, hybrid_cost, label='Hybrid: 5M + 0.15M/TB', linewidth=2, color='#8E44AD')
plt.plot(x, onprem_cost, label='On-premises: 10M + 0.1M/TB', linewidth=2, color='#27AE60')

# Draw minimum cost curve (Convex Hull)
plt.plot(x, min_costs, label='Minimum Cost (Convex Hull)', linewidth=3, color='#E74C3C', linestyle='--')

# Mark intersection points
plt.scatter([intersect1_x, intersect2_x], [intersect1_y, intersect2_y], color='#E74C3C', s=100, zorder=5)
plt.annotate(f'Intersection 1 ({intersect1_x}TB, {intersect1_y:.1f}M)',
             (intersect1_x, intersect1_y),
             xytext=(intersect1_x-15, intersect1_y+2),
             fontsize=12,
             color='#E74C3C')

plt.annotate(f'Intersection 2 ({intersect2_x}TB, {intersect2_y:.1f}M)',
             (intersect2_x, intersect2_y),
             xytext=(intersect2_x-25, intersect2_y+2),
             fontsize=12,
             color='#E74C3C')

# Mark optimal regions
plt.annotate('Cloud Optimal', (15, 10), fontsize=14, color='#E74C3C', weight='bold')
plt.annotate('Hybrid Optimal', (60, 15), fontsize=14, color='#E74C3C', weight='bold')
plt.annotate('On-premises Optimal', (85, 25), fontsize=14, color='#E74C3C', weight='bold')

# Mark optimal ranges on x-axis with arrows
plt.annotate('', xy=(0, -1), xytext=(intersect1_x, -1),
             arrowprops=dict(arrowstyle='<->', color='#3498DB', lw=2),
             xycoords=('data', 'axes fraction'),
             textcoords=('data', 'axes fraction'))

plt.annotate('', xy=(intersect1_x, -1), xytext=(intersect2_x, -1),
             arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2),
             xycoords=('data', 'axes fraction'),
             textcoords=('data', 'axes fraction'))

plt.annotate('', xy=(intersect2_x, -1), xytext=(100, -1),
             arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2),
             xycoords=('data', 'axes fraction'),
             textcoords=('data', 'axes fraction'))

# Graph settings
plt.title('Infrastructure Cost Optimization by Data Volume (Convex Hull Trick)', fontsize=18)
plt.xlabel('Monthly Data Volume (TB)', fontsize=14)
plt.ylabel('Monthly Total Cost (Million)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 100)
plt.ylim(0, 30)

# Adjust tick marks
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))

plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()