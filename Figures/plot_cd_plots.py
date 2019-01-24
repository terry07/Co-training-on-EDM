import Orange
import matplotlib.pyplot as plt
y = input('provide labeled ratio: (85 - 90 - 975 - super) ')


if y == '85':
		names = ["Co(Extra,GBC)","Co(NB,GBC)","Co(NB,Extra)","Co(RF,GBC)","Co(Extra,RF)","Co(GBC,GBC)","Co(GBC,Extra)","Co(RF,Extra)","Self(Extra)","CoForest"]
		avranks = [6.2875, 7.2625, 7.325, 7.525, 7.725, 8.5875, 8.7625, 9.0125, 9.0375, 12.625]
elif y == '90':
		names = ["Co(Extra,GBC)","Co(GBC,GBC)","Co(RF,GBC)","Co(NB,GBC)","Co(NB,Extra)","Co(Extra,RF)","Co(GBC,Extra)","Co(RF,RF)","Self(GBC)","CoForest"]
		avranks = [6.8875, 7.4125, 7.5375, 7.6125, 8.3, 8.6875, 9.2125, 9.2125, 9.925, 10.7125]
elif y == '975':
		names = ["Co(NB,Extra)","Co(Extra,Extra)","Co(Extra,RF)","Co(RF,Extra)","Co(GBC,GBC)","Co(Extra,GBC)","Co(NB,GBC)","Co(GBC,Extra)","Self(GBC)","CoForest"]
		avranks = [7.2875, 7.3, 7.9625, 8., 8.2, 8.25, 8.5625, 8.625, 9.9625, 12.5]
else:
		names = ["GBC","R = 15%: Co(Extra,GBC)", "RF", "Extra", "R = 10%: Co(Extra,GBC)", "R = 2.5%: Co(Extra,GBC)", "NB", "5NN"]
		avranks = [3.25, 3.7625, 4.2, 4.3125, 4.35, 4.575, 5.1375, 6.4125]
		

cd = Orange.evaluation.compute_CD(avranks, 40 , alpha = '0.05')
print("CD distance: ", cd)

plt.figure(figsize=(9, 9))

Orange.evaluation.graph_ranks(avranks, names, cd = cd, width = 15, textspace = 5)

plt.show()
plt.savefig('f1_score' + '_' + y + '.png', dpi=600)