import matplotlib.pyplot as plt

# Beispiel-Daten: Liste von FID50k-Scores
fid_scores = [506.90902799745794, 346.65978451872843, 460.00067171592747, 313.16237474025206, 186.5580560599991, 136.40217685602292, 106.20398530913471, 87.6352203246042, 78.78267132363987, 69.47890623655762, 64.05952986566466, 65.35959793308845, 63.51198068517615, 54.94383481939976, 50.56598058367004, 47.74191628810179, 47.784155567669465, 44.039220242319836, 41.25191248332096, 39.351703593008374, 36.65014871738524, 36.31055557833092, 35.88839231215866, 32.106286568120886, 31.885680554317744, 31.826770359353493, 29.624710227537182, 31.1857138015676, 28.74829515116282, 27.473520252159904, 27.82931323718346, 26.5344565617908, 25.35410515977808, 24.967158975564722, 25.428853479067566, 24.566292150821322, 23.17892758189656, 23.15730946966456, 23.17533407369129, 23.07744674608439]
epochs = list(range(1, len(fid_scores) + 1))

# Plot erstellen
plt.figure(figsize=(8, 6))
plt.plot(epochs, fid_scores, marker='o', linestyle='-', color='b', label='FID50k Score')
plt.title('FID50k Scores per Snapshot')
plt.xlabel('Snapshot')
plt.ylabel('FID50k Score')

# Show fewer x-ticks (only every 5th epoch)
plt.xticks(epochs[::5])

# Add annotations for every 5th value
for i in range(0, len(fid_scores), 5):
    plt.annotate(f'{fid_scores[i]:.2f}', 
                xy=(epochs[i], fid_scores[i]),
                xytext=(5, 5),
                textcoords='offset points',
                ha='left',
                va='bottom')

# Add text annotation for the last value
plt.annotate(f'FID: {fid_scores[-1]:.2f}', 
            xy=(epochs[-1], fid_scores[-1]),
            xytext=(5, 5),
            textcoords='offset points',
            ha='left',
            va='bottom')

plt.legend()
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid(True)
plt.show()
