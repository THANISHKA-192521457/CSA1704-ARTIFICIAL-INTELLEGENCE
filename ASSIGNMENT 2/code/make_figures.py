import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = '../results/'; F = '../figures/'
t1 = json.load(open(R+'task1_results.json')); t2 = json.load(open(R+'task2_results.json')); t3 = json.load(open(R+'task3_results.json'))
t4 = json.load(open(R+'task4_results.json')); t6 = json.load(open(R+'task6_results.json'))
# Fig 1: nodes expanded per map
maps = list(dict.fromkeys(r['map'] for r in t1['rows'])); import numpy as np
fig, ax = plt.subplots(figsize=(8.5, 3.6)); w = 0.26
for i, a in enumerate(('UCS', 'GBFS', 'ASTAR')):
    ax.bar(np.arange(len(maps))+(i-1)*w, [next(r['expanded'] for r in t1['rows'] if r['map'] == m and r['algo'] == a) for m in maps], w, label=a)
ax.set_xticks(range(len(maps))); ax.set_xticklabels([m.split('_')[0] for m in maps]); ax.set_ylabel('nodes expanded'); ax.legend(); ax.grid(axis='y', alpha=.3)
ax.set_title('Task 1: nodes expanded per map (A open, B cluttered, C maze, D trap, E mud)', fontsize=10); plt.tight_layout(); plt.savefig(F+'fig1_nodes.png', dpi=170); plt.close()
# Fig 2: cumulative moves over missions
c = t2['E5_missions']['cumulative_moves']; fig, ax = plt.subplots(figsize=(7, 3.6)); xs = range(1, 31)
for k, lab in (('dfs_every_time', 'online DFS every mission'), ('lrta_learning', 'LRTA* learning across missions'), ('map_once_then_astar', 'explore once, then reuse map (optimal path)')):
    ax.plot(xs, c[k], label=lab)
ax.set_xlabel('mission number (same facility)'); ax.set_ylabel('cumulative moves (mean of 20 mazes)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(F+'fig2_missions.png', dpi=170); plt.close()
# Fig 3: alpha-beta leaves
rows = {r['test']: r for r in t3['rows']}; names = ['best_ordered', 'equal_leaves', 'random_order_mean_of_30', 'worst_by_value_sort', 'engineered_no_cutoff']
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.bar(range(5), [rows[n]['ab_leaves'] for n in names], color=['#2a9d8f', '#2a9d8f', '#e9c46a', '#e76f51', '#9b2226'])
ax.set_yscale('log'); ax.axhline(65536, ls='--', c='k', lw=1); ax.text(0, 70000, 'plain minimax: 65,536 leaves', fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(['best\nordered', 'equal\nleaves', 'random\n(mean 30)', 'worst by\nvalue sort', 'engineered\nno cut-off'], fontsize=8)
ax.set_ylabel('leaves evaluated (log)'); ax.set_title('Task 3: alpha-beta work, b=4, d=8', fontsize=10); ax.grid(axis='y', alpha=.3)
for i, n in enumerate(names): ax.text(i, rows[n]['ab_leaves']*1.15, f"{rows[n]['ab_leaves']:,}", ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(F+'fig3_alphabeta.png', dpi=170); plt.close()
# Fig 4: minimax growth
fig, ax = plt.subplots(figsize=(7, 3.6))
for bd, mk in ((5, 'o'), (6, 's'), (7, '^')):
    g = [r for r in t4['growth'] if r['b'] == bd]
    ax.semilogy([r['depth'] for r in g], [r['nodes'] for r in g], mk+'-', label=f'b={bd} observed')
    ax.semilogy([r['depth'] for r in g], [r['theory_nodes'] for r in g], 'k:', lw=.8)
ax.set_xlabel('lookahead depth d'); ax.set_ylabel('positions examined (log)'); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title('Task 4: observed (markers) vs (b^(d+1)-1)/(b-1) (dotted)', fontsize=10)
plt.tight_layout(); plt.savefig(F+'fig4_minimax_growth.png', dpi=170); plt.close()
# Fig 6: exhaustive growth
g = t6['growth']; fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].semilogy([r['n'] for r in g], [r['tours_enumerated'] for r in g], 'o-', label='tours enumerated'); ax[0].semilogy([r['n'] for r in g], [r['predicted_tours'] for r in g], 'k:', label='(n-1)!')
ax[0].set_xlabel('customers + depot n'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].semilogy([r['n'] for r in g], [r['time_s'] for r in g], 's-'); ax[1].axhline(2, ls='--', c='r'); ax[1].text(5, 2.4, '2 s response limit', color='r', fontsize=8)
ax[1].set_xlabel('n'); ax[1].set_ylabel('exhaustive time (s)'); ax[1].grid(alpha=.3); plt.tight_layout(); plt.savefig(F+'fig6_tsp_growth.png', dpi=170); plt.close()
# Fig 7: redesign
t7 = json.load(open(R+'task7_results.json')); c = t7['comparison']; fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.semilogy([r['n'] for r in c], [r['exhaustive_calls'] for r in c], 'o-', label='exhaustive (original)')
for k, lab in (('R1_prefix_only', 'R1 prefix cut'), ('R1R2_bound', 'R1+R2 bound'), ('R1R2R3_bound_order', 'R1+R2+R3 ordering'), ('R1R2R3R4_full', 'R1-R4 full redesign')):
    ax.semilogy([r['n'] for r in c], [r[k]['calls'] for r in c], 's-', label=lab)
ax.set_xlabel('n'); ax.set_ylabel('recursive calls (log)'); ax.legend(fontsize=8); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(F+'fig7_redesign.png', dpi=170); plt.close()
print('figures done')
