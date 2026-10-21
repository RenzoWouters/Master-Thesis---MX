"""Create a NEW overview copy with actual completed E2M1 data. No training.
Only standard-library Python is needed. Never overwrites either input folder.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
from dedicated_data import load_dedicated,FILES,require

def ascode(text):return {'id':hashlib.sha256(('code'+text).encode()).hexdigest()[:12],'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':text.splitlines(True)}
def asmd(text):return {'id':hashlib.sha256(('md'+text).encode()).hexdigest()[:12],'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}

def integrate(run_dir, overview_dir, output_dir, *, _synthetic_test=False):
    run_dir, overview_dir, output_dir=map(lambda p:Path(p).resolve(),(run_dir,overview_dir,output_dir))
    data=load_dedicated(run_dir,allow_synthetic=_synthetic_test)
    require((overview_dir/'bronmanifest.json').is_file() and (overview_dir/'meeting_data.py').is_file(), 'Gebruik de volledig uitgepakte overzichtsmap v1.')
    require(not output_dir.exists(), 'Uitvoermap bestaat al: kies een NIEUWE map.')
    require(output_dir not in run_dir.parents and run_dir not in output_dir.parents and output_dir!=run_dir,'Uitvoer mag niet in/over de runmap staan.')
    require(overview_dir not in output_dir.parents and output_dir not in overview_dir.parents and output_dir!=overview_dir,'Uitvoer moet naast, niet in, het oude overzicht staan.')
    notebook=overview_dir/'MX_onderzoeksoverzicht.ipynb'
    n=json.loads(notebook.read_text(encoding='utf-8'))
    require(len(n['cells'])==37 and '6. Het specifieke voordeel' in ''.join(n['cells'][17]['source']), 'Niet het verwachte overzicht v1; niets aanpassen.')
    shutil.copytree(overview_dir,output_dir)
    dest=output_dir/'bronnen/dedicated_e2m1_100';dest.mkdir(exist_ok=False)
    for name in FILES:shutil.copy2(run_dir/name,dest/name)
    shutil.copy2(Path(__file__).with_name('dedicated_data.py'),output_dir/'dedicated_data.py')
    (output_dir/'DEDICATED_INTEGRATIE.json').write_text(json.dumps({'source_run':str(run_dir), 'source_overview':str(overview_dir),
        'synthetic_only':data['synthetic'], 'files_sha256':data['files_sha256'],
        'note':'Original overview sources unchanged; new results are added from a completed run.'},indent=2),encoding='utf-8')
    def get(i):return ''.join(n['cells'][i]['source'])
    def put(i,s):n['cells'][i]['source']=s.splitlines(True)
    put(1,get(1)+f'''\nimport dedicated_data as DD
DDATA = DD.load_dedicated(BASE / 'bronnen/dedicated_e2m1_100', allow_synthetic={_synthetic_test!r})
DED = pd.DataFrame(DDATA['metrics'])
for col in ['epoch','n','correct1']: DED[col] = pd.to_numeric(DED[col]).astype(int)
for col in ['count_top1_percent','objective_loss','lr']: DED[col] = pd.to_numeric(DED[col])
DED['accuracy_percent'] = DED['count_top1_percent']
DED_LAST = DED.query('epoch == 100 and phase == "test"').iloc[0]
if DDATA['synthetic']: note('## SYNTHE TISCHE VALIDATIE: dedicated datapunt is GEEN onderzoeksresultaat.'.replace('SYNTHE TISCHE','SYNTHETISCHE'))
assert (DDATA['code_scale_bytes']+S['auxiliary_bytes']) == 88608
''')
    # Add dedicated to the existing protocol and endpoint tables, retaining old rows.
    x=get(6).replace('tabel(pd.DataFrame(protocol))',"protocol.append({'Training':'Dedicated E2M1/A32','Modi':'4','Activaties':'A32 vast','Forward/backward per batch':1,'Updates':DDATA['report']['optimizer_steps'],'Epochs':100,'Lage pad':'direct E2M1; geen hoge modus'})\ntabel(pd.DataFrame(protocol))")
    x=x.replace("tabel(x, {'Accuracy (%)':'{:.4f}', 'Test-CE':'{:.4f}'})", "x=pd.concat([x,pd.DataFrame([{'Training':'Dedicated E2M1/A32','Uitvoering':'Directe E2M1','Accuracy (%)':DED_LAST.accuracy_percent,'Test-CE':DED_LAST.objective_loss}])],ignore_index=True)\ntabel(x, {'Accuracy (%)':'{:.4f}', 'Test-CE':'{:.4f}'})")
    put(6,x)
    put(7,get(7).replace('Voor **een dedicated E2M1/A32-model ontbreekt nog een training**. Daarover staat bewust geen verzonnen datapunt in de figuren.', 'De nieuwe dedicated E2M1-referentie staat nu in de tabellen en figuren; ook daarvan is maar een seed onderzocht.'))
    x=get(12).replace("for e in [50,75,90]: ax.axvline(e,linestyle=':',alpha=0.35)\nax.set(xlabel='Afgeronde epoch',", "qd=DED.query('phase == \"test\"')\nax.plot(qd.epoch,qd.accuracy_percent,label='Dedicated E2M1/A32',linewidth=1.4)\nfor e in [50,75,90]: ax.axvline(e,linestyle=':',alpha=0.35)\nax.set(xlabel='Afgeronde epoch',")
    x=x.replace("title='Lage modus: directe kwantisatie tegenover truncatiebewust',ylim=(86,96)", "title='Lage modus: dedicated, gezamenlijk direct en truncatiebewust',ylim=(min(86,float(qd.accuracy_percent.min())-0.5),max(96,float(qd.accuracy_percent.max())+0.3))")
    put(12,x)
    put(17,get(17)+'''\n\n**Dit zijn vier opslagstrategieen, geen vier trainingen.** De derde balk zijn twee zelfstandige exports van hetzelfde gezamenlijke model, niet twee dedicated trainingen. De tweede balk deelt alleen gelijk bevonden schalen/bijgegevens, maar behoudt beide BN-banken en beide codesets. De vierde bewaart een hoge bron voor beide modi. Heb je slechts E2M1 nodig, dan is een vaste vierbitskopie kleiner; dat is geen flexibele bron voor twee modi.''')
    put(18,get(18).replace('Twee zelfstandige exports/modellen','Twee exports van gezamenlijk model').replace('een E2M1-dedicated accuracy ontbreekt. De opslagvorm is berekenbaar, maar de bijbehorende kwaliteitsvergelijking nog niet.', 'de dedicated E2M1-accuracy is nu gemeten. De derde balk blijft echter gebaseerd op twee exports van hetzelfde gezamenlijke model en is geen geoptimaliseerde dedicated-opslag.'))
    x=get(22).replace("for rid,label,mark in", "ax.scatter([S['view_low_all_bytes']/1024],[DED_LAST.accuracy_percent],marker='v',s=85,label='E2M1 afzonderlijk (nieuw)')\nfor rid,label,mark in")
    x=x.replace("ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=0.25)", "lo=min(93.3,float(DED_LAST.accuracy_percent)-0.2);hi=max(95.3,float(DED_LAST.accuracy_percent)+0.2)\nax.set_ylim(lo,hi)\nax.legend(loc='lower right',fontsize=9);ax.grid(alpha=0.25)")
    x=x.replace("T=TIJD.copy();", "T=pd.concat([TIJD,pd.DataFrame([{'methode':'Dedicated E2M1/A32','passes_per_batch':1,'epoch_total_s':DDATA['epoch_seconds'],'session_s':DDATA['report'].get('session_wall_seconds',DDATA['epoch_seconds'])}])],ignore_index=True);")
    put(22,x)
    put(23,get(23).replace('een passende dedicated E2M1/A32-training ontbreekt. Niet nodig om huidige resultaten te tonen, wel om het vierbits-accuracyverschil tegenover dedicated te beoordelen.', 'de dedicated E2M1/A32-training is toegevoegd. Het is een vergelijking van trainingsregimes (loss, aantal passes en distillatie verschillen), geen afzonderlijk causaal distillatie-effect. Herhalingen blijven nodig.'))
    for c in n['cells']:
        if c['cell_type']=='code':c['outputs']=[];c['execution_count']=None
    newcells=[asmd('## 2b. Dedicated E2M1 tegenover de drie lage uitvoeringen\nVast vergelijkingspunt: epoch100. Rechtstreeks dedicated E2M1 is geen truncatie. Het aantal optimizerupdates is gelijk; gezamenlijke modellen gebruiken meer passes en een ander leerdoel.'),ascode('''d=DED_LAST
q=M.final_row(D,'dual100',4);tr=M.final_row(D,'trunc100',4);post=M.gap_row(D,'C_truncate_E2M3')
comp=pd.DataFrame([
 {'Training / uitvoering':'Dedicated / direct E2M1','Accuracy (%)':d.accuracy_percent,'Test-CE':d.objective_loss},
 {'Training / uitvoering':'Gezamenlijk direct / direct E2M1','Accuracy (%)':q.accuracy_percent,'Test-CE':q.objective_loss},
 {'Training / uitvoering':'Gezamenlijk direct / achteraf trunceren','Accuracy (%)':post.accuracy_percent,'Test-CE':post.hard_cross_entropy},
 {'Training / uitvoering':'Gezamenlijk truncatiebewust / trunceren','Accuracy (%)':tr.accuracy_percent,'Test-CE':tr.objective_loss}])
comp['Verschil met dedicated (procentpunt)']=comp['Accuracy (%)']-d.accuracy_percent
tabel(comp,{'Accuracy (%)':'{:.4f}','Test-CE':'{:.4f}','Verschil met dedicated (procentpunt)':'{:+.4f}'})
sel=pd.DataFrame(DDATA['selected'])
tabel(sel[['role','epoch','correct1','n','count_top1_percent']])
init=[]
for rid in ['single_mx100','dual100','trunc100']:
 h=D['histories'][rid] if 'histories' in D else None
 init.append({'Referentie':rid,'Initialisatie gelijk':DDATA['report']['initial_model_sha256']==D['reports'][rid].get('initial_model_sha256'),
              'Notitie':'Zelfde seed is geen bewijs van identiek GPU-pad; samplevolgorde afzonderlijk controleren.'})
tabel(pd.DataFrame(init))
comp.to_csv(BASE/'afgeleid/dedicated_E2M1_vergelijking.csv',index=False)
bron(['dedicated_e2m1_100'],'epoch_metrics.csv')
''')]
    n['cells'][8:8]=newcells
    (output_dir/'MX_onderzoeksoverzicht.ipynb').write_text(json.dumps(n,indent=1,ensure_ascii=False),encoding='utf-8')
    # Do not leave a stale rendered overview that looks updated but is not.
    (output_dir/'MX_onderzoeksoverzicht.html').unlink(missing_ok=True)
    (output_dir/'README_E2M1.md').write_text('Nieuwe kopie met echte dedicated E2M1-data. Open MX_onderzoeksoverzicht.ipynb en Run All in .venv-train. De oude HTML is niet mee overgenomen omdat zij deze nieuwe resultaten niet bevat. Alle bronbestanden van het oorspronkelijke overzicht blijven byte-identiek. Dedicated-versus-joint is geen iso-compute/iso-loss experiment; een seed per methode. Geen hardwaremetingen.\n',encoding='utf-8')
    return output_dir

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--overview-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        out=integrate(args.run_dir,args.overview_dir,args.output_dir)
    except (ValueError,FileNotFoundError,KeyError) as e:
        ap.exit(1,'Geen geldig overzicht gemaakt: '+str(e)+'\n')
    print('NIEUWE OVERZICHTSMAP:',out)
    print('Open MX_onderzoeksoverzicht.ipynb uit die map en kies Run All. Geen training uitgevoerd.')
if __name__=='__main__':main()
