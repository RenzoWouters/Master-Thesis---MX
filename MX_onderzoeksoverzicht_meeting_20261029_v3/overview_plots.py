"""Presentation helpers embedded verbatim in the standalone meeting notebook.

Only uses the already loaded DATA object and numpy/pandas/matplotlib. No torch,
datasets, checkpoints, downloads, or training operations.
"""

RUNS = {r['id']: r for r in DATA['training_runs']}
POST = DATA['posthoc']['cases']
COST = DATA['costs']
FORMAT = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
FORMAT_LONG = {8: 'MXFP8 · E4M3', 6: 'MXFP6 · E2M3', 4: 'MXFP4 · E2M1'}
COLORS = {8: '#25658A', 6: '#248776', 4: '#C77B25'}
INK, MUTED, GRID = '#19384B', '#60727E', '#E1E7EC'
RUN_LABELS = {
    'github_anyprecision5': 'GitHub Any-Precision · 5 modi',
    'dedicated_w32_control': 'Dedicated W32/A32-controle',
    'dedicated_mxfp6': 'Dedicated E2M3',
    'dedicated_mxfp4': 'Dedicated E2M1',
    'dedicated_e4m3': 'Dedicated E4M3',
    'dual_direct': 'Twee modi · direct',
    'dual_mantissa': 'Twee modi · mantissatruncatie',
    'triple_direct': 'Drie modi · direct',
    'triple_naive': 'Exponent + mantissa · schaal ×1',
    'triple_scale64': 'Exponent + mantissa · schaal ×64',
    'triple_scale32': 'Exponent + mantissa · schaal ×32',
    'triple_learned100': 'Keuze ×32/×64 leren tot epoch 100',
    'triple_freeze50': 'Keuze leren tot 50 · daarna vast',
    'triple_group16': 'Groepgrootte 16 · schaal ×64',
    'triple_group64': 'Groepgrootte 64 · schaal ×64',
    'dedicated': 'Dedicated per formaat (aparte runs)',
}
DEDICATED = {8: 'dedicated_e4m3', 6: 'dedicated_mxfp6', 4: 'dedicated_mxfp4'}
TRIPLE_IDS = ['dedicated', 'triple_direct', 'triple_naive', 'triple_scale64',
              'triple_scale32', 'triple_learned100', 'triple_freeze50',
              'triple_group16', 'triple_group64']
_FIGURES, _PLOT_INPUTS = {}, {}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#B8C6CE', 'axes.labelcolor': INK,
    'axes.titlecolor': INK, 'axes.titleweight': 'bold',
    'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'figure.dpi': 110,
    'axes.grid': False, 'legend.frameon': False,
})


def dec(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')


def accuracy(run_id, bit, role='epoch100'):
    rid = DEDICATED[bit] if run_id == 'dedicated' else run_id
    run = RUNS[rid]
    rows = run['epoch100'] if role == 'epoch100' else run['selected']['rows']
    found = [r for r in rows if int(r['weight_bits']) == bit]
    assert len(found) == 1, (rid, bit, role)
    return float(found[0]['top1_percent'])


def posthoc(suite, parent, path, role='epoch100'):
    rows = [r for r in POST if r['suite_id'] == suite and
            r['training_variant_id'] == parent and r['path'] == path and
            r['checkpoint_role'] == role]
    assert len(rows) == 1, (suite, parent, path, role, len(rows))
    return rows[0]


def postacc(suite, parent, path, role='epoch100'):
    r = posthoc(suite, parent, path, role)
    return float(r['reported_accuracy_percent'])


def caption(fig, text, y=.012):
    fig.text(.015, y, text, ha='left', va='bottom', fontsize=9, color=MUTED)


def title(fig, main, subtitle=''):
    fig.suptitle(main, x=.015, y=.985, ha='left', fontsize=17, fontweight='bold')
    if subtitle:
        fig.text(.015, .935, subtitle, ha='left', va='top', color=MUTED, fontsize=10)


def register(name, fig, values=None):
    assert name not in _FIGURES, f'Dubbele figuurnaam: {name}'
    _FIGURES[name] = fig
    _PLOT_INPUTS[name] = values if values is not None else []
    display(fig)
    plt.close(fig)
    # Do not return a Figure: IPython would display it again as the cell result.
    return None


def table(frame, title_text=None, force=False):
    if SHOW_DETAIL_TABLES or force:
        if title_text:
            display(Markdown('**' + title_text + '**'))
        display(frame)


def checkpoint_table(ids, role='epoch100', bits=(8, 6, 4)):
    out = []
    for rid in ids:
        row = {'Training': RUN_LABELS.get(rid, rid)}
        if rid == 'dedicated':
            row['Epoch'] = '100 / 100 / 100' if role == 'epoch100' else ' / '.join(
                str(RUNS[DEDICATED[b]]['selected']['epoch']) for b in bits)
        else:
            row['Epoch'] = 100 if role == 'epoch100' else RUNS[rid]['selected']['epoch']
        for bit in bits:
            row[FORMAT[bit] + ' [%]'] = round(accuracy(rid, bit, role), 6)
        out.append(row)
    return pd.DataFrame(out)


def github_plot(role='epoch100', name='01_github_epoch100'):
    bits = [1, 2, 4, 8, 32]
    vals = [accuracy('github_anyprecision5', b, role) for b in bits]
    epoch = 100 if role == 'epoch100' else RUNS['github_anyprecision5']['selected']['epoch']
    fig, ax = plt.subplots(figsize=(10, 4.9))
    title(fig, f'GitHub Any-Precision · CNN-8/SVHN · epoch {epoch}',
          'Vijf gekoppelde gewichts- én activatieprecisies; SVHN train-only')
    fig.subplots_adjust(top=.77, bottom=.19, left=.085, right=.98)
    xs = np.arange(5)
    ax.bar(xs, vals, width=.6, color=['#9BAEBE', '#7195B0', '#477C9F', '#25658A', '#19384B'])
    for x, v in zip(xs, vals):
        ax.text(x, v + 1.3, dec(v) + '%', ha='center', fontsize=10)
    ax.set(xticks=xs, xticklabels=[f'W{b}/A{b}' for b in bits], ylim=(0, 103),
           ylabel='Testaccuracy [%]')
    ax.set_yticks(range(0, 101, 20)); ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
    caption(fig, 'Oorspronkelijke GitHub-quantisator; eigen run met train zonder extra. MX-proeven hierna houden activaties op A32.')
    return register(name, fig, [{'bits': b, 'accuracy': v, 'epoch': epoch} for b, v in zip(bits, vals)])


def dual_plot(role='epoch100', name='02_twee_MX_epoch100'):
    ids = ['dedicated', 'dual_direct', 'dual_mantissa']
    labels = ['Dedicated per formaat\n(twee aparte runs)', 'Gezamenlijk\nrechtstreeks quantiseren',
              'Gezamenlijk\nmantissatruncatie']
    fig, ax = plt.subplots(figsize=(11, 5.5))
    title(fig, 'Twee MX-precisies · ' + ('epoch 100' if role == 'epoch100' else 'gekozen hoge-precisiecheckpoints'),
          'MXFP6/E2M3 en MXFP4/E2M1 · A32 · groepgrootte 32')
    fig.subplots_adjust(top=.76, bottom=.24, left=.09, right=.98)
    records = []
    for b, shift in [(6, -.14), (4, .14)]:
        vals = [accuracy(rid, b, role) for rid in ids]
        xs = np.arange(3) + shift
        ax.scatter(xs, vals, s=85, color=COLORS[b], marker='o' if b == 6 else 's', label=FORMAT_LONG[b], zorder=3)
        for x, v, rid in zip(xs, vals, ids):
            ax.annotate(dec(v), (x, v), xytext=(-7 if b == 6 else 7, 11 if b == 6 else -20),
                        textcoords='offset points', ha='right' if b == 6 else 'left', fontsize=10, color=COLORS[b])
            records.append({'training': rid, 'bits': b, 'role': role, 'accuracy': v})
    ax.set(xticks=np.arange(3), xticklabels=labels, ylabel='Testaccuracy [%] · ingezoomde as', ylim=(94.65, 95.30), xlim=(-.5, 2.55))
    ax.grid(axis='y', color=GRID); ax.legend(loc='lower left', fontsize=9)
    caption(fig, 'Dedicated: één hard-CE-pass. Gezamenlijk: twee passes, recursieve distillatie en één Adam-update per minibatch.')
    return register(name, fig, records)


def triple_plot(role='epoch100', detail=False, name=None):
    ids = [r for r in TRIPLE_IDS if not(detail and r == 'triple_naive')]
    name = name or ('03_triple_epoch100_detail' if detail else '03_triple_epoch100')
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.0), sharey=True)
    endpoint = 'epoch 100' if role == 'epoch100' else 'checkpoint gekozen op hoogste E4M3'
    title(fig, 'Drie MX-precisies · ' + endpoint,
          'Elke kolom toont hetzelfde uitvoerformaat; dedicated bestaat uit drie afzonderlijke trainingen' +
          (' · detail zonder de naïeve ×1-variant' if detail else ''))
    fig.subplots_adjust(left=.29, right=.98, top=.84, bottom=.13, wspace=.14)
    ys = np.arange(len(ids)); records = []
    labels = []
    for rid in ids:
        label = RUN_LABELS[rid]
        if role != 'epoch100':
            ep = '/'.join(str(RUNS[DEDICATED[b]]['selected']['epoch']) for b in (8, 6, 4)) if rid == 'dedicated' else str(RUNS[rid]['selected']['epoch'])
            label += f'  [e{ep}]'
        labels.append(label)
    low, high = ((93.5, 95.38) if detail else (84.8, 96.1)) if role == 'epoch100' else (85.0, 96.2)
    for ax, b in zip(axes, (8, 6, 4)):
        for i in range(len(ids)):
            if i % 2 == 0: ax.axhspan(i-.44, i+.44, color='#F3F6F8', zorder=0)
        vals = [accuracy(rid, b, role) for rid in ids]
        ax.scatter(vals, ys, color=COLORS[b], s=63, zorder=3)
        for rid, y, v in zip(ids, ys, vals):
            ax.annotate(dec(v), (v, y), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9, color=COLORS[b])
            records.append({'training': rid, 'bits': b, 'role': role, 'accuracy': v})
        ax.set(title=FORMAT_LONG[b], xlim=(low, high), xlabel='Testaccuracy [%] · zoom')
        ax.grid(axis='x', color=GRID)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(ys, labels); axes[0].set_ylim(len(ids)-.5, -.5)
    caption(fig, 'Alle MX-modi gebruiken A32 en FP32-masters. Groepgrootte 32, behalve waar 16 of 64 staat. Eén seed per traject.')
    return register(name, fig, records)


def dot_rows(name, main, subtitle, labels, vals, colors=None, limits=None, note=''):
    fig, ax = plt.subplots(figsize=(12, max(4.8, 2.3 + .52*len(labels))))
    title(fig, main, subtitle)
    fig.subplots_adjust(left=.35, right=.97, top=.78, bottom=.17)
    ys = np.arange(len(labels)); colors = colors or [COLORS[6]]*len(vals)
    for i in range(len(labels)):
        if i % 2 == 0: ax.axhspan(i-.42, i+.42, color='#F3F6F8', zorder=0)
    ax.scatter(vals, ys, c=colors, s=77, zorder=3)
    for y, v, c in zip(ys, vals, colors):
        ax.annotate(dec(v), (v, y), xytext=(7, 0), textcoords='offset points', va='center', color=c)
    if limits is None: limits = (min(vals)-.15, max(vals)+.3)
    ax.set(yticks=ys, yticklabels=labels, ylim=(len(labels)-.5, -.5), xlim=limits,
           xlabel='Testaccuracy [%] · ingezoomde as')
    ax.grid(axis='x', color=GRID); ax.tick_params(axis='y', length=0)
    caption(fig, note)
    return register(name, fig, [{'label': l, 'accuracy': v} for l, v in zip(labels, vals)])


def mx_posthoc_plot():
    labels = ['GitHub W32/A32 · bron vóór conversie', 'Dezelfde bron · MXFP6 achteraf',
              'Dedicated W32/A32 · aparte controle', 'Dedicated MXFP6 · MX tijdens training',
              'Twee modi direct · E2M3', 'Twee modi met mantissatruncatie · E2M3']
    vals = [postacc('mx_cast_anyprecision','github_anyprecision5','reference_W32_A32'),
            postacc('mx_cast_anyprecision','github_anyprecision5','MXFP6_E2M3_A32'),
            accuracy('dedicated_w32_control',32), accuracy('dedicated_mxfp6',6),
            accuracy('dual_direct',6), accuracy('dual_mantissa',6)]
    return dot_rows('04_MX_achteraf_en_training', 'MX-quantisatie · achteraf en tijdens training',
                    'Alle eindpunten: epoch 100 · W32/A32-bron of E2M3/A32-uitvoering', labels, vals,
                    colors=[MUTED, '#9975AA', MUTED, COLORS[6], COLORS[6], COLORS[6]], limits=(94.4,95.3),
                    note='Alleen de eerste twee rijen delen hetzelfde checkpoint. Andere rijen zijn aparte trainingen met andere losses/passes.')


def mantissa_plot(role='epoch100', name='05_mantissa_achteraf_en_training'):
    paths = ['A_direct_E2M1','B_round_from_E2M3','C_truncate_E2M3']
    vals = [postacc('dual_mantissa','dual_direct',p,role) for p in paths]
    vals.append(accuracy('dual_mantissa',4,'epoch100' if role=='epoch100' else 'selected'))
    return dot_rows(name, 'Mantissatruncatie · wat verandert na aanpassing tijdens training?',
                    'E2M1/A32 · ' + ('epoch 100' if role=='epoch100' else 'oorspronkelijke E2M3-selecties'),
                    ['A · direct E2M1 uit master', 'B · E2M3 → E2M1 afronden', 'C · E2M3 → E2M1 bits afkappen',
                     'Afzonderlijk truncatiebewust getraind'], vals,
                    colors=[MUTED,'#9975AA','#B95A51',COLORS[4]], limits=(93.25,95.45),
                    note='A/B/C: hetzelfde direct getrainde checkpoint en dezelfde bevroren BN-bank. Laatste rij: nieuwe training, BN leert mee.')


def triple_posthoc_plot(role='epoch100', name='06_triple_alle_bitbewerkingen'):
    paths = ['direct8','direct6','direct4','round64','mantissa64','round86','exponent86','combined864']
    labels = ['Direct E4M3','Direct E2M3','Direct E2M1','E2M3 → E2M1 afronden',
              'E2M3 → E2M1 mantissatruncatie','E4M3 → E2M3 afronden · S behouden',
              'E4M3 → E2M3 exponenttruncatie · S behouden', 'E4M3 → E2M3 → E2M1 · S behouden']
    rows = [posthoc('triple_initial','triple_direct',p,role) for p in paths]
    vals = [float(r['reported_accuracy_percent']) for r in rows]
    fig, ax = plt.subplots(figsize=(13,6.8))
    title(fig, f'Bitbewerkingen achteraf · één triple-direct-checkpoint · epoch {rows[0]["epoch"]}',
          'Mantissa-, exponent- en gecombineerde routes op exact dezelfde getrainde gewichten')
    fig.subplots_adjust(left=.37,right=.97,top=.81,bottom=.16)
    ys=np.arange(len(labels)); cols=[COLORS[int(r['weight_bits'])] for r in rows]
    ax.barh(ys,vals,color=cols,height=.64)
    for y,v in zip(ys,vals): ax.text(v+1,y,dec(v),va='center',fontsize=9)
    ax.set(yticks=ys,yticklabels=labels,ylim=(len(labels)-.5,-.5),xlim=(0,105),xlabel='Testaccuracy [%]')
    ax.grid(axis='x',color=GRID);ax.set_axisbelow(True)
    caption(fig,'S is telkens de schaal van de gekozen bron. A32; vaste doel-BN-bank; geen BN-herkalibratie of optimizerupdates.')
    return register(name,fig,[{'case_id':r['case_id'],'accuracy':v} for r,v in zip(rows,vals)])


def exponent_adaptation_plot(role='epoch100', name='07_exponent_achteraf_en_training'):
    trainrole='epoch100' if role=='epoch100' else 'selected'
    labels=['Direct gezamenlijk getraind', 'Naïeve exponentroute achteraf · ×1',
            'Naïeve exponentroute tijdens training · ×1', 'Exponentroute achteraf · ×64',
            'Exponentroute tijdens training · ×64']
    fig, axes=plt.subplots(1,2,figsize=(14,6.3),sharey=True)
    title(fig,'Exponenttruncatie · achteraf versus tijdens training',
          ('Epoch 100 van elke brontraining' if role=='epoch100' else 'Oorspronkelijke hoogste-precisiecheckpoints')+' · daarna telkens dezelfde bedoelde uitvoerroute')
    fig.subplots_adjust(left=.32,right=.97,top=.79,bottom=.16,wspace=.18)
    ys=np.arange(5); records=[]
    for ax,b in zip(axes,[6,4]):
        vals=[accuracy('triple_direct',b,trainrole),
              postacc('triple_scale_controls','triple_direct','A'+str(b),role),
              accuracy('triple_naive',b,trainrole),
              postacc('triple_scale_controls','triple_direct','B'+str(b),role),
              accuracy('triple_scale64',b,trainrole)]
        bars=ax.barh(ys,vals,color=[MUTED,'#C7B6D2',COLORS[b],'#C7B6D2',COLORS[b]],height=.65)
        for i in (1,3): bars[i].set_hatch('//'); bars[i].set_edgecolor('#8C709B')
        for y,v in zip(ys,vals):ax.text(v+1,y,dec(v),va='center',fontsize=9)
        ax.set(title=FORMAT_LONG[b],xlim=(0,108),xlabel='Testaccuracy [%]')
        ax.grid(axis='x',color=GRID);ax.set_axisbelow(True)
        records.extend({'label':l,'bits':b,'accuracy':v,'role':role} for l,v in zip(labels,vals))
    axes[0].set_yticks(ys,labels);axes[0].set_ylim(4.5,-.5)
    caption(fig,'Gearceerd = achteraf op het direct getrainde model. Trainingsrijen hebben een eigen model, inclusief aangepaste BN-statistieken.')
    return register(name,fig,records)


def scale_control_plot(parents=('triple_direct','triple_naive'), suite='triple_scale_controls',
                       role='epoch100', name='08_schaal_en_afronding'):
    fig,axes=plt.subplots(1,len(parents),figsize=(14 if len(parents)>1 else 10,5.6),squeeze=False,sharey=True)
    title(fig,'Schaal en afronding afzonderlijk gecontroleerd',
          'A: exponentbits + S · B: dezelfde bits + 64S · C: afronden uit gedecodeerde E4M3-bron + 64S')
    fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.22,wspace=.15)
    records=[]
    for ax,parent in zip(axes[0],parents):
        for b,offset in [(6,-.18),(4,.18)]:
            paths=[fam+str(b) for fam in ['A','B','C']]
            vals=[postacc(suite,parent,p,role) for p in paths]
            xs=np.arange(3)+offset
            ax.bar(xs,vals,width=.32,color=COLORS[b],label=FORMAT[b])
            for x,v,p in zip(xs,vals,paths):
                ax.text(x,v+1.1,dec(v,2),ha='center',fontsize=9)
                records.append({'parent':parent,'path':p,'role':role,'accuracy':v})
        epoch=posthoc(suite,parent,'A6',role)['epoch']
        ax.set(title=RUN_LABELS[parent]+f' · e{epoch}',xticks=range(3),xticklabels=['A · schaal ×1','B · schaal ×64','C · afronden, ×64'],ylim=(0,105),ylabel='Testaccuracy [%]')
        ax.grid(axis='y',color=GRID);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9)
    caption(fig,'Binnen elk paneel: hetzelfde checkpoint en bevroren BN. C4 wordt door mantissatruncatie van C6 afgeleid; geen FP32-masterafronding.')
    return register(name,fig,records)


def comparison_curves(ids, name, main, start=1, end=100, lower=None):
    fig,axes=plt.subplots(1,3,figsize=(15,5.3),sharey=True)
    title(fig,main,f'Echte testmetingen na elke epoch · epochs {start}–{end} · geen smoothing')
    fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.25,wspace=.12)
    pal=['#25658A','#C77B25','#248776','#9A65AD','#697B86','#BA554A']
    records=[]
    for ax,b in zip(axes,[8,6,4]):
        for rid,c in zip(ids,pal):
            rows=sorted([r for r in RUNS[rid]['test_history'] if int(r['weight_bits'])==b and start<=r['epoch']<=end],key=lambda r:r['epoch'])
            ax.plot([r['epoch'] for r in rows],[r['top1_percent'] for r in rows],color=c,lw=1.7,label=RUN_LABELS[rid])
            records.extend({'training':rid,'bits':b,'epoch':r['epoch'],'accuracy':r['top1_percent']} for r in rows)
        for boundary in [50,75,90]:
            if start<=boundary<=end:ax.axvline(boundary+.5,color='#85949E',ls=':',lw=.9)
        ax.set(title=FORMAT_LONG[b],xlabel='Afgeronde epoch',xlim=(start,end))
        ax.grid(color=GRID)
        if lower is not None:ax.set_ylim(lower,95.35)
    axes[0].set_ylabel('Testaccuracy [%] · ingezoomde as')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.065),ncol=min(3,len(ids)),fontsize=9)
    caption(fig,'Stippellijnen: LR-wijziging na 50/75/90. Epochs van één run zijn onderling afhankelijke metingen, geen onafhankelijke herhalingen.')
    return register(name,fig,records)


def scale_learning_plot():
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    title(fig,'Geleerde schaalkeuze · dynamisch versus vast na epoch 50',
          '4.080 groepen · één gedeelde keuze ×32 of ×64 voor beide lagere modi')
    fig.subplots_adjust(left=.08,right=.98,top=.77,bottom=.23,wspace=.22)
    records=[]
    for rid,color in [('triple_learned100','#9A65AD'),('triple_freeze50','#248776')]:
        rows=RUNS[rid]['scale_history']; xs=[r['epoch'] for r in rows]
        axes[0].plot(xs,[100*r['fraction_x32'] for r in rows],color=color,label=RUN_LABELS[rid],lw=1.9)
        axes[1].plot(xs,[r['choice_flips_during_epoch'] for r in rows],color=color,label=RUN_LABELS[rid],lw=1.5)
        records.extend({'training':rid,**r} for r in rows)
    for ax in axes:
        ax.axvline(50.5,color=MUTED,ls='--');ax.set(xlabel='Afgeronde epoch',xlim=(1,100));ax.grid(color=GRID)
    axes[0].set(ylabel='Groepen met ×32 [%]',ylim=(-1,51));axes[1].set_ylabel('Keuzeomslagen per epoch')
    axes[1].ticklabel_format(style='plain',axis='y')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=2,fontsize=9)
    caption(fig,'Een groep kan meermaals omslaan. Freeze houdt de keuze vast; de E4M3-bronschaal blijft van de actuele gewichten afhangen.')
    return register('09_geleerde_schaalkeuzes',fig,records)


def group_checkpoint_plot():
    ids=['triple_group16','triple_scale64','triple_group64']; names=['B16','B32','B64']
    palette=['#248776','#C77B25','#25658A']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.7),sharex=True,sharey=True)
    title(fig,'Groepsgrootte · beide checkpointkeuzes', 'Alle varianten gebruiken vaste ×64 · drie scores per variant uit hetzelfde checkpoint')
    fig.subplots_adjust(left=.08,right=.98,top=.73,bottom=.23,wspace=.16)
    records=[]
    for ax,role,lab in zip(axes,['epoch100','selected'],['Epoch 100','Checkpoint gekozen op E4M3']):
        for j,(rid,c,label) in enumerate(zip(ids,palette,names)):
            vals=[accuracy(rid,b,role) for b in [8,6,4]];ys=np.arange(3)+(j-1)*.2
            ax.scatter(vals,ys,color=c,s=61,marker=['o','s','D'][j],label=label,zorder=3)
            for b,y,v in zip([8,6,4],ys,vals):
                ax.annotate(dec(v),(v,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=9,color=c)
                records.append({'training':rid,'bits':b,'role':role,'accuracy':v})
        ep='B16:53 · B32:42 · B64:53' if role=='selected' else 'B16:100 · B32:100 · B64:100'
        ax.set(title=lab+'\n'+ep,xlim=(93.65,95.35),xlabel='Testaccuracy [%] · zoom',yticks=range(3),yticklabels=['E4M3','E2M3','E2M1'],ylim=(2.55,-.55))
        ax.grid(axis='x',color=GRID)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3)
    caption(fig,'B32 kiest vóór de eerste LR-daling; B16/B64 erna. Op hetzelfde epoch 53 scoort B32 hoger in alle drie de modi.')
    return register('11_groepsgrootte_checkpoints',fig,records)


def active_storage_plot():
    ids=['fp32','mx_e4m3_b32','mx_e2m3_b32','mx_e2m1_b32']
    rows=[next(r for r in COST['active_layouts'] if r['id']==i) for i in ids]
    labels=['FP32','E4M3 · B32','E2M3 · B32','E2M1 · B32']
    fig,ax=plt.subplots(figsize=(11.5,5.6));title(fig,'Opslag van één actieve modelkopie',
        'Gepakte code-inhoud + gedeelde schalen + alle bestaande FP32/BN-bijstaat; geen container-overhead')
    fig.subplots_adjust(left=.16,right=.97,top=.77,bottom=.24)
    y=np.arange(len(rows)); left=np.zeros(len(rows))
    for field,label,c in [('code_bytes','Gewichtscodes','#25658A'),('scale_bytes','Shared scales','#C77B25'),('aux_bytes','FP32/BN-bijstaat','#AEBCC5')]:
        vals=np.array([r[field]/1024 for r in rows]);ax.barh(y,vals,left=left,label=label,color=c,height=.62);left+=vals
    for yi,r,total in zip(y,rows,left): ax.text(total+6,yi,dec(total,2)+' KiB',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=labels,ylim=(3.6,-.6),xlim=(0,max(left)*1.18),xlabel='Logische modelinhoud [KiB; 1 KiB = 1.024 bytes]')
    ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.legend(loc='lower center',bbox_to_anchor=(.47,-.35),ncol=3,fontsize=9)
    caption(fig,'129.472 doelgewichten; 19.248 bytes bijstaat, inclusief alle vijf BN-banken. Een actieve lage kopie is iets anders dan een flexibele bron.')
    return register('12_actieve_modelopslag',fig,rows)


def group_storage_plot():
    rows=[r for r in COST['active_layouts'] if r['id']!='fp32']
    fig,axes=plt.subplots(1,3,figsize=(14.3,5.7),sharey=True)
    title(fig,'Groepsgrootte · schaaloverhead versus padding',
          'Extra bytes boven de echte gewichtscodes; de constante FP32/BN-bijstaat staat buiten deze overheadgrafiek')
    fig.subplots_adjust(left=.08,right=.98,top=.77,bottom=.23,wspace=.18)
    for ax,b in zip(axes,[8,6,4]):
        rs=sorted([r for r in rows if r['weight_bits']==b],key=lambda r:r['group_size'])
        pad=np.array([r['padding_slots']*b/8/1024 for r in rs]); scales=np.array([r['scale_bytes']/1024 for r in rs]);xs=np.arange(3)
        ax.bar(xs,pad,color='#AEBCC5',label='Paddingcodes',width=.58)
        ax.bar(xs,scales,bottom=pad,color='#C77B25',label='Shared scales',width=.58)
        for x,p,s in zip(xs,pad,scales):ax.text(x,p+s+.12,dec(p+s,2),ha='center',fontsize=9)
        ax.set(title=FORMAT_LONG[b],xticks=xs,xticklabels=['B16','B32','B64'],ylabel='Overhead [KiB]',ylim=(0,12))
        ax.grid(axis='y',color=GRID);ax.set_axisbelow(True)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.075),ncol=2)
    caption(fig,'Kleinere groepen: meer schaalbytes. Grotere groepen: hier meer rijpadding. Voor deze laagvormen is B32 netto het kleinst.')
    return register('13_groepsgrootte_opslag',fig,rows)


def flexible_storage_plot():
    rows=COST['flexible_sources']
    fig,axes=plt.subplots(1,2,figsize=(16,8.6))
    title(fig,'Eén compacte bron voor meerdere uitvoerprecisies',
          'Vergelijking bij dezelfde functie: twee modi beschikbaar (links), drie modi beschikbaar (rechts)')
    fig.subplots_adjust(left=.17,right=.98,top=.83,bottom=.14,wspace=1.05)
    records=[]
    for ax,num in zip(axes,[2,3]):
        ids=(['dual_dedicated_separate','dual_direct_shared_aux_scales','dual_mantissa_source'] if num==2 else
             ['triple_dedicated_separate','triple_direct_shared_aux','triple_naive_source',
              'triple_scale64_source','triple_scale32_source','triple_learned100_source',
              'triple_freeze50_source','triple_group16_source','triple_group64_source'])
        rs=[next(r for r in rows if r['id']==key) for key in ids]
        labels=(['Twee dedicated modellen','Direct · gedeelde bijstaat\nen gelijke schalen','Eén E2M3-bron\nmet mantissatruncatie'] if num==2 else
                ['Drie dedicated modellen','Direct · drie codesets\nen gedeelde bijstaat','E4M3-bron · naïef ×1',
                 'E4M3-bron · vast ×64','E4M3-bron · vast ×32','E4M3-bron · geleerd tot 100',
                 'E4M3-bron · geleerd tot 50','E4M3-bron · B16, ×64','E4M3-bron · B64, ×64'])
        ys=np.arange(len(rs));left=np.zeros(len(rs))
        for field,label,c in [('code_bytes','Gewichtscodes','#25658A'),('scale_bytes','Schalen','#C77B25'),('choice_bytes','Keuzebits','#9A65AD'),('aux_bytes','Bijstaat','#AEBCC5')]:
            vals=np.array([r[field]/1024 for r in rs]);ax.barh(ys,vals,left=left,height=.64,label=label,color=c);left+=vals
        for y,total in zip(ys,left):ax.text(total+4,y,dec(total,1),va='center',fontsize=9)
        ax.set(title=f'{num} modi beschikbaar',yticks=ys,yticklabels=labels,ylim=(len(rs)-.5,-.5),xlim=(0,max(left)*1.2),xlabel='Logische broninhoud [KiB]')
        ax.tick_params(axis='y',labelsize=9);ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);records+=rs
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.52,.055),ncol=4,fontsize=9)
    caption(fig,'Directe codebundels met gedeelde bijstaat zijn layoutscenario’s. Truncatiebronnen zijn geëxporteerd; geleerde keuzes vragen 510 extra bytes.')
    return register('14_flexibele_bronopslag',fig,records)


def training_storage_plot():
    rows=COST['training_state']; fig,ax=plt.subplots(figsize=(13,6.0))
    title(fig,'Trainingsstaat blijft FP32', 'Modelstaat + Adam-momenten; gradients apart zichtbaar, overige werkgeheugens uitgesloten')
    fig.subplots_adjust(left=.30,right=.97,top=.77,bottom=.24)
    ys=np.arange(len(rows));left=np.zeros(len(rows))
    for field,label,c in [('base_state_bytes','Basis-modelstaat','#AEBCC5'),('theta_bytes','FP32-schaalparameters','#9A65AD'),('adam_moment_bytes','Adam-momenten','#25658A'),('gradient_bytes','Gradients','#6FB0A4')]:
        vals=np.array([r[field]/1024**2 for r in rows]);ax.barh(ys,vals,left=left,color=c,height=.62,label=label);left+=vals
    for y,t in zip(ys,left):ax.text(t+.035,y,dec(t,3),va='center',fontsize=9)
    ax.set(yticks=ys,yticklabels=[r['label'] for r in rows],ylim=(len(rows)-.5,-.5),xlim=(0,max(left)*1.16),xlabel='Geïnventariseerde tensorinhoud [MiB]')
    ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.legend(loc='lower center',bbox_to_anchor=(.42,-.35),ncol=2,fontsize=9)
    caption(fig,'Geen activaties, cache, werkbuffers, allocator of Adam-stepcounters. Freeze gebruikt hier lr=0 en behoudt theta-gradients en Adam-momenten.')
    return register('15_FP32_trainingsstaat',fig,rows)


def timing_plot():
    ids=[r['id'] for r in DATA['training_runs']]
    vals=[RUNS[r]['timing']['epoch_total_seconds']/60 for r in ids]
    fig,ax=plt.subplots(figsize=(13,9));title(fig,'Gemeten trainingsduur · alle volledige trajecten',
        'Som van de opgeslagen epochmetingen, inclusief test en checkpointwerk; afzonderlijke laptopuitvoeringen')
    fig.subplots_adjust(left=.35,right=.97,top=.85,bottom=.13)
    ys=np.arange(len(ids));cols=['#9A65AD' if 'learned' in r or 'freeze' in r else '#25658A' for r in ids]
    ax.barh(ys,vals,color=cols,height=.65)
    records=[]
    for y,r,v in zip(ys,ids,vals):
        passes=RUNS[r]['timing']['training_passes'];p=int(round(passes/57300))
        ax.text(v+1,y,f'{dec(v,1)} min · {p} pass'+('es' if p!=1 else '')+'/batch',va='center',fontsize=9)
        records.append({'training':r,'minutes':v,'training_passes':passes})
    ax.set(yticks=ys,yticklabels=[RUN_LABELS[r] for r in ids],ylim=(len(ids)-.5,-.5),xlim=(0,max(vals)*1.35),xlabel='Duur [minuten]')
    ax.grid(axis='x',color=GRID);ax.set_axisbelow(True)
    caption(fig,'Freeze: duur van geërfde epochs 1–50 + nieuwe epochs 51–100. Gelijk aantal updates betekent niet hetzelfde aantal passes of rekentijd.')
    return register('16_gemeten_trainingsduur',fig,records)


def all_checkpoint_table(role='epoch100'):
    records=[]
    for rid,r in RUNS.items():
        rows=r['epoch100'] if role=='epoch100' else r['selected']['rows']
        for q in rows:
            b=int(q['weight_bits'])
            label=f'W{b}/A{b}' if rid=='github_anyprecision5' else FORMAT_LONG.get(b,'W32/A32')
            records.append({'Training':RUN_LABELS[rid],'Modus':label,'Epoch':100 if role=='epoch100' else r['selected']['epoch'],
                            'Accuracy [%]':round(q['top1_percent'],6),'Correct':q['correct1'],'Testbeelden':q['n'],
                            'Selectieregel':r['selected']['criterion']})
    return pd.DataFrame(records)


def protocol_table():
    rows=[]
    for rid,r in RUNS.items():
        c=r['config'];t=r['timing'];bits=r['weight_bits']
        rows.append({'Training':RUN_LABELS[rid],'Modi':', '.join(str(b) for b in sorted(bits,reverse=True)),
                     'Activaties':r['activation_policy'],'Groepgrootte':r['group_size'],
                     'Passes per batch':round(t['training_passes']/57300),'Updates':t['optimizer_steps'],
                     'Schaalkeuze geleerd':c.get('learned_shared_scale',False),
                     'Selectie-epoch':r['selected']['epoch']})
    return pd.DataFrame(rows)


def posthoc_table(role=None):
    rows=[]
    for r in POST:
        if role and r['checkpoint_role']!=role:continue
        rows.append({'Proef':r['suite_id'],'Brontraining':RUN_LABELS[r['training_variant_id']],
                     'Rol':r['checkpoint_role'],'Epoch':r['epoch'],'Route':r['path_label'],
                     'Formaat':r['weight_format'],'Accuracy [%]':round(r['reported_accuracy_percent'],6),
                     'Correct':r['correct1'],'BN herkalibreerd':r['batchnorm_recalibrated'],
                     'Bronbestand':r['source_metrics_file']})
    return pd.DataFrame(rows)


def source_table():
    rows=[]
    for rid,r in RUNS.items():
        s=r['source'];rows.append({'Training':RUN_LABELS[rid],'Oorspronkelijke run':r['run_id'],
                                 'Feedbackbundel':s['bundle_name'],'Metrics':s['metrics_path'],
                                 'SHA-256 metrics':s['metrics_sha256']})
    return pd.DataFrame(rows)


def validate_embedded_data():
    assert len(RUNS)==15
    assert sum(len(r['epoch100']) for r in RUNS.values())==37
    assert len(POST)==90
    for rid,r in RUNS.items():
        assert r['timing']['optimizer_steps']==57300
        for b in r['weight_bits']:
            hist=sorted([q for q in r['test_history'] if int(q['weight_bits'])==int(b)],key=lambda q:q['epoch'])
            assert [q['epoch'] for q in hist]==list(range(1,101)),(rid,b)
        for q in r['epoch100']+r['selected']['rows']+r['test_history']:
            assert q['n']==26032
            assert abs(q['top1_percent']-100*q['correct1']/q['n'])<1e-5
        high=max(r['weight_bits']); hist=[q for q in r['test_history'] if int(q['weight_bits'])==high]
        expected=max(hist,key=lambda q:(q['top1_percent'],-q['epoch']))['epoch']
        assert r['selected']['epoch']==expected,(rid,expected)
        for b in r['weight_bits']:
            q=next(q for q in r['test_history'] if q['epoch']==100 and q['weight_bits']==b)
            assert accuracy(rid,b)==q['top1_percent']
    for r in POST:
        assert r['n']==26032 and r['optimizer_steps']==0 and r['batchnorm_recalibrated'] is False
        assert abs(r['reported_accuracy_percent']-100*r['correct1']/26032)<1e-5
    for b in [8,6,4]:
        a=[q for q in RUNS['triple_learned100']['test_history'] if q['epoch']<=50 and q['weight_bits']==b]
        f=[q for q in RUNS['triple_freeze50']['test_history'] if q['epoch']<=50 and q['weight_bits']==b]
        assert a==f
    for r in COST['active_layouts']:
        assert r['total_model_bytes']==r['code_bytes']+r['scale_bytes']+r['aux_bytes']
    for r in COST['flexible_sources']:
        assert r['total_bytes']==sum(r[k] for k in ['code_bytes','scale_bytes','choice_bytes','aux_bytes'])
    return {'Trajecten':len(RUNS),'Epoch-100-metingen':37,'Testmetingen tijdens training':sum(len(r['test_history']) for r in RUNS.values()),
            'Posthoc-metingen':len(POST),'Datacontrole':'PASS'}


def export_figures():
    if not SAVE_FIGURES:
        print(f'{len(_FIGURES)} figuren getoond. SAVE_FIGURES=False: geen bestanden geschreven.')
        return None
    destination=Path.cwd()/('MX_meeting_figuren_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(exist_ok=False)
    for name,fig in _FIGURES.items():
        fig.savefig(destination/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(destination/(name+'.svg'),bbox_inches='tight')
    (destination/'grafiekgegevens.json').write_text(json.dumps(_PLOT_INPUTS,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Figuren en bijbehorende meetwaarden opgeslagen in:',destination)
    return destination
