import matplotlib.pyplot as plt
import numpy as np

# Results
methods = ['Centralized\nBaseline', 'Federated\n(No DP)', 'Federated\n(With DP ε=50)']
accuracies = [73.55, 63.69, 39.21]
colors = ['#2196F3', '#4CAF50', '#FF9800']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Federated Learning Framework — Results Summary', fontsize=16, fontweight='bold')

# Bar chart
bars = ax1.bar(methods, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
ax1.set_ylim(0, 100)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Accuracy Comparison', fontsize=13)
ax1.axhline(y=73.55, color='#2196F3', linestyle='--', alpha=0.4, label='Centralized baseline')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Privacy-Utility tradeoff
privacy_levels = ['No Privacy\n(Centralized)', 'Basic Privacy\n(Federated)', 'Strong Privacy\n(Federated+DP)']
privacy_scores = [0, 50, 90]
accuracy_scores = [73.55, 63.69, 39.21]

ax2.plot(privacy_scores, accuracy_scores, 'o-', color='#9C27B0', linewidth=2.5, markersize=10)
for i, (p, a, label) in enumerate(zip(privacy_scores, accuracy_scores, privacy_levels)):
    ax2.annotate(f'{label}\n({a}%)', (p, a),
                textcoords="offset points", xytext=(10, -15),
                fontsize=9, color='#333333')
ax2.set_xlabel('Privacy Level →', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Privacy vs Accuracy Tradeoff', fontsize=13)
ax2.set_xlim(-10, 105)
ax2.set_ylim(30, 85)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Graph saved to results/accuracy_comparison.png")
plt.show()