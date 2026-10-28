"""Presentation helpers for already-produced transfer results; never runs models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KIB=1024
MIB=1024**2
ORDER=['github5','dual_direct','dual_mantissa','triple_direct','triple_scale64_b32','triple_scale64_b64']
LABELS={'github5':'GitHub · 5 precisies','dual_direct':'2 MX · direct · B32',
        'dual_mantissa':'2 MX · mantissatruncatie · B32','triple_direct':'3 MX · direct · B32',
        'triple_scale64_b32':'3 MX · truncatie ×64 · B32',
        'triple_scale64_b64':'3 MX · truncatie ×64 · B64'}
COLORS=dict(zip(ORDER,['#637682','#4E8B80','#BA873E','#416A8D','#9174A0','#AE685E']))


def find_results(result_dir=None):
    if result_dir is not None:
        p=Path(result_dir).expanduser().resolve()
        path=p if p.name=='run_results.json' else p/'run_results.json'
        if not path.is_file(): raise FileNotFoundError(str(path))
        return path
    cwd=Path.cwd()
    if (cwd/'run_results.json').is_file(): return cwd/'run_results.json'
    found={}
    for root in [cwd,*list(cwd.parents)[:5]]:
        parent=root/'mx_runs'/'mx_precision_switch_transfer_eval'
        if parent.is_dir():
            for path in parent.glob('*/run_results.json'):
                try:
                    data=json.loads(path.read_text(encoding='utf-8-sig'))
                    if data.get('status')=='PASS': found[str(path.resolve())]=path
                except (OSError,ValueError): pass
    if not found:
        raise FileNotFoundError('Geen brongeverifieerde run_results.json gevonden. Voer eerst run_check.py run uit, of vul RESULT_DIR expliciet in.')
    return max(found.values(),key=lambda p:p.stat().st_mtime)


def load_results(result_dir=None):
    path=find_results(result_dir)
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema')!='mx_precision_switch_transfer_results_v1':
        raise ValueError('Onbekend resultaatformaat')
    names=('storage','initial_loading','mode_payloads','summary','trace','activations')
    tables={name:pd.DataFrame(data[name]) for name in names}
    accuracy_path=path.parent/'data'/'accuracy_context.json'
    accuracy=json.loads(accuracy_path.read_text(encoding='utf-8-sig')) if accuracy_path.is_file() else []
    rows=[]
    for model in accuracy:
        for row in model['epoch100']:
            rows.append(dict(model_id=model['model_id'],model=LABELS[model['model_id']],
                             run_id=model['run_id'],**row))
    tables['accuracy']=pd.DataFrame(rows)
    return path.parent,data,tables


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#C3CCD1','axes.labelcolor':'#263D49',
                         'xtick.color':'#435965','ytick.color':'#435965',
                         'text.color':'#263D49','figure.facecolor':'white',
                         'axes.facecolor':'white','savefig.facecolor':'white'})


def _primary(df):
    return df[df['primary_variant']].copy() if 'primary_variant' in df else df.copy()


def _select(tables,cache_kib=128,line_bytes=64,preload_policy='none',scenario='alternating_extremes',primary=True):
    df=tables['summary']
    if primary: df=_primary(df)
    return df[(df.cache_kib==cache_kib)&(df.line_bytes==line_bytes)&
              (df.preload_policy==preload_policy)&(df.scenario==scenario)].copy()


def _ordered(df):
    return df.set_index('model_id').reindex(ORDER)


def _grid(ax,axis='x'):
    ax.grid(axis=axis,color='#E4E9EC',linewidth=.8); ax.set_axisbelow(True)


def render(fig,name,outdir=None):
    from IPython.display import display
    if outdir is not None:
        outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True)
        fig.savefig(outdir/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(outdir/(name+'.svg'),bbox_inches='tight')
    display(fig);plt.close(fig)


def storage_plot(tables):
    df=_ordered(_primary(tables['storage']))
    fig,ax=plt.subplots(figsize=(11,5.1));left=np.zeros(len(ORDER))
    for column,color,label in [('codes','#416A8D','Gewichtscodes'),('scales','#BA873E','Schalen'),('aux','#A4AFB6','Overige inferenceparameters')]:
        values=df[column].to_numpy()/KIB
        ax.barh(np.arange(len(ORDER)),values,left=left,color=color,label=label,height=.62)
        left+=values
    ax.set_yticks(range(len(ORDER)),[LABELS[x] for x in ORDER]);ax.invert_yaxis()
    for i,value in enumerate(left): ax.text(value+max(left)*.012,i,f'{value:.1f}',va='center',fontsize=9)
    ax.set_xlim(0,max(left)*1.16);ax.set_xlabel('Compacte operationele opslag [KiB]')
    ax.set_title('Alle ondersteunde modi beschikbaar houden',loc='left',pad=18)
    _grid(ax);ax.legend(loc='lower right',fontsize=9)
    fig.text(.02,.015,'Voorgestelde indeling; alleen actieve BN-banken. Geen checkpointbestandsgrootte of trainingsgeheugen.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.05,1,1));return fig


def loading_plot(tables):
    df=_primary(tables['initial_loading']);fig,axes=plt.subplots(1,3,figsize=(16.5,5.8),sharey=True)
    for ax,which,title in zip(axes,['high','low','ALL'],['Alleen hoogste modus','Alleen laagste modus','Alle modi één keer lezen']):
        selected=[]
        for mid in ORDER:
            own=df[df.model_id==mid]
            modes=[int(x) for x in own['mode'] if str(x)!='ALL']
            mode=str(max(modes) if which=='high' else min(modes)) if which!='ALL' else 'ALL'
            selected.append(own[own['mode']==mode].iloc[0])
        selected=pd.DataFrame(selected);left=np.zeros(len(ORDER))
        for col,color,label in [('codes_read_bytes','#416A8D','Codes'),('scales_read_bytes','#BA873E','Schalen'),('aux_read_bytes','#A4AFB6','Overig')]:
            values=selected[col].to_numpy()/KIB
            ax.barh(range(len(ORDER)),values,left=left,color=color,label=label,height=.65);left+=values
        for i,value in enumerate(left): ax.text(value+max(left)*.015,i,f'{value:.1f}',va='center',fontsize=8.5)
        ax.set_xlim(0,max(left)*1.22);ax.set_title(title,loc='left',fontsize=12)
        ax.set_xlabel('Koude laadtransfers [KiB]');_grid(ax)
    axes[0].set_yticks(range(len(ORDER)),[LABELS[x] for x in ORDER]);axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right',fontsize=8.5)
    fig.suptitle('Initiële loading apart: startmodus of alle operationele modidata',x=.02,ha='left',fontsize=15)
    line=int(df.line_bytes.iloc[0])
    fig.text(.02,.015,f'Hele {line}-byte-lijnen; lege cache. Panelen hebben een eigen x-schaal. Hoog/laag: GitHub W32/W1, twee MX W6/W4, drie MX W8/W4.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.055,1,.94));return fig


def switching_plot(tables,cache_kib=128,line_bytes=64):
    fig,axes=plt.subplots(1,2,figsize=(15,5.8),sharey=True)
    xmax=max(_select(tables,cache_kib,line_bytes,p).weight_total_bytes.max() for p in ['none','all_regions'])/MIB*1.15
    for ax,preload,title in zip(axes,['none','all_regions'],['Op aanvraag: eerste inference is initiële loading','Alle modi vooraf lezen door dezelfde cache']):
        df=_ordered(_select(tables,cache_kib,line_bytes,preload));x=df.initial_load_bytes.to_numpy()/MIB;y=df.continuation_bytes.to_numpy()/MIB
        ax.barh(range(len(ORDER)),x,color='#BA873E',label='Initiële loading',height=.64)
        ax.barh(range(len(ORDER)),y,left=x,color='#416A8D',label='Vervolg',height=.64)
        for i,value in enumerate(x+y):ax.text(value+max(x+y)*.012,i,f'{value:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,xmax);ax.set_title(title,loc='left',fontsize=11.5)
        ax.set_xlabel('Parametertransfers voor volledige reeks [MiB]');_grid(ax)
    axes[0].set_yticks(range(len(ORDER)),[LABELS[x] for x in ORDER]);axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right',fontsize=9)
    fig.suptitle(f'Precisiewissels · hoogste ↔ laagste · {cache_kib} KiB parametercache',x=.02,ha='left',fontsize=15)
    fig.text(.02,.015,'Modelberekening; truncatie met selectieve codedelen. Initieel + vervolg = totaal. Voorladen garandeert bij een kleine cache geen residentie.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.06,1,.94));return fig


def cache_plot(tables,line_bytes=64):
    df=_primary(tables['summary']);df=df[(df.line_bytes==line_bytes)&(df.scenario=='alternating_extremes')&(df.preload_policy=='none')]
    capacities=sorted(df.cache_kib.unique());fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for mid in ORDER:
        rows=df[df.model_id==mid].set_index('cache_kib').reindex(capacities)
        for ax,col,factor in [(axes[0],'avg_weight_bytes_per_inference',1/KIB),(axes[1],'weight_hit_ratio',100)]:
            ax.plot(range(len(capacities)),rows[col]*factor,'o-',color=COLORS[mid],label=LABELS[mid],linewidth=1.8,markersize=4)
    for ax in axes:
        ax.set_xticks(range(len(capacities)),[str(x) for x in capacities]);ax.set_xlabel('Parametercache [KiB] · discrete onderzochte capaciteiten');_grid(ax,'y')
    axes[0].set_ylabel('Gemiddelde parametertransfers [KiB/inference]');axes[1].set_ylabel('Cachehitratio [% van aangevraagde lijnen]');axes[1].set_ylim(-2,102)
    axes[0].set_title('Verkeer inclusief koude start',loc='left');axes[1].set_title('Hergebruik bij precisiewissels',loc='left')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8.7)
    fig.suptitle('Het effect van beschikbare bufferruimte',x=.02,ha='left',fontsize=15)
    fig.tight_layout(rect=(0,.14,1,.94));return fig


def cumulative_plot(tables,cache_kib=128,line_bytes=64):
    df=_primary(tables['trace']);df=df[(df.cache_kib==cache_kib)&(df.line_bytes==line_bytes)&(df.preload_policy=='none')&(df.scenario=='alternating_extremes')]
    panels=[('GitHub',['github5']),('Twee MX-modi',['dual_direct','dual_mantissa']),('Drie MX-modi',['triple_direct','triple_scale64_b32','triple_scale64_b64'])]
    fig,axes=plt.subplots(1,3,figsize=(16,5.1))
    for ax,(title,ids) in zip(axes,panels):
        for mid in ids:
            rows=df[df.model_id==mid].sort_values('index')
            ax.step(rows['index']+1,rows.cumulative_weight_bytes/MIB,where='post',label=LABELS[mid],color=COLORS[mid],linewidth=2)
        ax.axvspan(.5,1.5,color='#BA873E',alpha=.2,label='Eerste inference / initiële loading')
        ax.set_title(title,loc='left');ax.set_xlabel('Inference in de wisselreeks');ax.set_ylabel('Cumulatieve parametertransfers [MiB]');_grid(ax,'y')
        ax.legend(loc='upper left',fontsize=7.8)
    fig.suptitle(f'Wanneer worden nieuwe gegevens opgehaald? · {cache_kib} KiB · op aanvraag',x=.02,ha='left',fontsize=15)
    fig.text(.02,.012,'Panelen hebben een eigen verticale schaal. Activaties worden afzonderlijk getoond; elke inference heeft een nieuwe input.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.055,1,.94));return fig


def full_source_plot(tables,line_bytes=64):
    ids=['dual_mantissa','triple_scale64_b32','triple_scale64_b64'];fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
    for ax,cache in zip(axes,[0,128]):
        df=_select(tables,cache,line_bytes,primary=False)
        for i,(variant,color,label) in enumerate([('selective','#4E8B80','Selectieve codedelen'),('full_source','#AE685E','Volledige bron opvragen')]):
            rows=df[(df.model_id.isin(ids))&(df.variant==variant)].set_index('model_id').reindex(ids)
            values=rows.weight_total_bytes.to_numpy()/MIB
            bars=ax.barh(np.arange(len(ids))+(i-.5)*.3,values,height=.27,color=color,label=label)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_title(f'{cache} KiB parametercache',loc='left');ax.set_xlabel('Parametertransfers voor volledige reeks [MiB]');_grid(ax)
        ax.margins(x=.18)
    axes[0].set_yticks(range(len(ids)),[LABELS[x] for x in ids]);axes[0].invert_yaxis()
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,fontsize=9)
    fig.suptitle('Zelfde truncatiemodel, twee opslag-/uitleesvarianten',x=.02,ha='left',fontsize=15)
    fig.text(.02,.012,'Allebei behouden exact dezelfde bitroute. De volledige bron kan bij voldoende cachecapaciteit ook worden hergebruikt.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.16,1,.94));return fig


def activation_plot(tables,line_bytes=64):
    df=tables['activations'];df=df[df.line_bytes==line_bytes];fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    github=df[df.model_id=='github5'].sort_values('mode',ascending=False)
    mx=df[df.model_id!='github5'].drop_duplicates('model_id').set_index('model_id').reindex(ORDER[1:])
    for ax,rows,labels,title in [(axes[0],github,[f'W{int(x)}/A{int(x)}' for x in github['mode']],'GitHub · gekoppelde W/A-modi'),(axes[1],mx,[LABELS[x]+' · A32' for x in ORDER[1:]],'MX · A32 bij iedere gewichtsmodus')]:
        read=rows.activation_read_bytes.to_numpy()/KIB;write=rows.activation_write_bytes.to_numpy()/KIB
        ax.barh(range(len(rows)),read,color='#416A8D',label='Lezen',height=.63)
        ax.barh(range(len(rows)),write,left=read,color='#BA873E',label='Schrijven',height=.63)
        for i,value in enumerate(read+write):ax.text(value+2,i,f'{value:.1f}',va='center',fontsize=9)
        ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,150);ax.set_title(title,loc='left',fontsize=12)
        ax.set_xlabel('Activatietransfers per inference [KiB]');_grid(ax)
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,fontsize=9)
    fig.suptitle('Activatieverkeer afzonderlijk van de parameters',x=.02,ha='left',fontsize=15)
    fig.text(.02,.012,'3×40×40-input; samengevoegde conv/BN/pool/activatiefasen. Compacte lage GitHub-activaties zijn een opslagscenario.',fontsize=9,color='#6F8089')
    fig.tight_layout(rect=(0,.16,1,.94));return fig


def line_plot(tables,cache_kib=64):
    df=_primary(tables['summary']);df=df[(df.cache_kib==cache_kib)&(df.scenario=='alternating_extremes')&(df.preload_policy=='none')]
    lines=sorted(df.line_bytes.unique());fig,ax=plt.subplots(figsize=(12,5.8))
    for i,line in enumerate(lines):
        rows=_ordered(df[df.line_bytes==line]);values=rows.weight_total_bytes.to_numpy()/MIB
        ax.barh(np.arange(len(ORDER))+(i-(len(lines)-1)/2)*.23,values,height=.21,
                label=f'{line} bytes per transfer',color=['#7BA297','#416A8D','#BA873E'][i%3])
    ax.set_yticks(range(len(ORDER)),[LABELS[x] for x in ORDER]);ax.invert_yaxis();_grid(ax)
    ax.set_xlabel('Parametertransfers voor volledige wisselreeks [MiB]')
    ax.set_title(f'Gevoeligheid voor transfergrootte · {cache_kib} KiB parametercache',loc='left',pad=16)
    ax.legend(loc='lower right',fontsize=9);fig.tight_layout();return fig


def scenario_table(tables,cache_kib=128,line_bytes=64):
    df=_primary(tables['summary']);df=df[(df.cache_kib==cache_kib)&(df.line_bytes==line_bytes)&(df.preload_policy=='none')].copy()
    df['Model']=df.model_id.map(LABELS)
    df['Parameterverkeer [MiB]']=df.weight_total_bytes/MIB
    df['Activatieverkeer [MiB]']=df.activation_total_bytes/MIB
    return df[['Model','scenario','n_inferences','n_switches','Parameterverkeer [MiB]','Activatieverkeer [MiB]']].reset_index(drop=True)
