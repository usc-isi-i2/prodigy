"""Private, figure-led contribution decision; not a manuscript expansion."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter


def main():
    root=Path(__file__).parent
    repo=next(p for p in root.parents if (p/'AGENTS.md').is_file())
    out=repo/'output/pdf/class_reference_contribution_decision.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=17,leading=20,textColor=HexColor('#173b55'),spaceAfter=5))
    styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=9.2,leading=11.5,spaceAfter=7))
    styles.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=7.7,leading=9.4,textColor=HexColor('#555555'),spaceAfter=5))
    sections=[
        ('Explanation established',
         'Support context can damage the class reference without changing the query being classified. In Hong Kong-to-political transfer, replacing support values improves ranking with attention routes and final query vectors exactly fixed (B-C). The effect does not require different attention weights or query information. This localizes the failure to transported support content; it does not yet explain which natural contexts will be harmful.'),
        ('Generality result',
         'The nine-source map makes Hong Kong an extreme case, not the only source with harmful support context (A). The more specific value-over-key prediction survives the nominated 50k configuration in both streams: +6.86/+8.93 versus +0.15/+0.84 within-episode AUC points. These are new intervention outcomes in an already-known second configuration, not a held-out-source mechanism test.'),
        ('Strongest counterexample',
         'The useful original-style Wiki-to-FB15K-237 model needs both roles: native accuracy 73.80% falls to 39.93% with support suppression and 39.00% with query suppression. Moreover, political suppression at 50k improves ranking but reduces accuracy to about 25%. Hence neither universal harmful support processing nor deployable suppression is supported. Observed cross-role coupling in the public model prevents the fixed-query isolation used in the social model.'),
        ('What the recent tests changed',
         'The late Hong Kong model learns a receive-edge task well on fixed graph inputs (AUC .91/.92), improves a content-feature task, and remains better with 40% balanced support exceptions. Original bot rankings still deteriorate among nonzero-biography queries. These results rule against the tested capability-loss stories; they do not connect the bot trajectory causally to political reference damage.'),
        ('Contribution and decision',
         '<b>Transported support content can change discrimination even when attention and queries do not change.</b> Keep this empirical dissociation as the paper center, rather than restating the architecture. Independent review identifies the gap as transfer of this pathway explanation beyond one source-target pair, not a universal sign predictor. One nominated natural source-target K/V test could address it. Close the synthetic branch; the current evidence is a bounded causal diagnosis, not yet the intended high-impact explanation.')]
    sections = [
        ('Explanation', 'Graph context changes what supports contribute to the class reference, not only which supports receive attention. Political ranking improves when support values from the edge-suppressed condition replace native values while attention and queries remain exactly fixed (B-C). The same pathway carries useful bot information: value replacement hurts, while key replacement helps (D-E). This localizes a causal pathway; why its content helps or hurts remains unresolved.'),
        ('Generality', 'The independently nominated Ukraine-to-TwiBot20 test changes both source and task. Its frozen value-over-key ordering passes each stream: values -6.42/-4.98 versus keys +1.19/+0.93 AUC points. Bot accuracy also falls with values (-4.72/-3.19 points). The political ordering previously survived 50k training (B). The nine-source map establishes broader role heterogeneity, not nine-source K/V mediation (A).'),
        ('Strongest counterexample', 'The original-style public Wiki-to-FB15K-237 model needs both roles: 73.80% accuracy falls to 39.93%/39.00% under support/query suppression. Its coupled computation does not permit the same fixed-query isolation. At political 50k, better rankings coexist with accuracy falling to about 25%. Suppression is therefore neither a general repair nor a deployment method.'),
        ('Contribution and decision', '<b>Context utility depends on the content transported into the class reference, even at fixed attention and query representation.</b> Diagnose this computation separately from encoder quality and routing. Pursue the scoped mechanism paper; stop the synthetic rule/noise branch and do not launch another sweep. The nominated pathway test now succeeds beyond the discovery pair, but significance beyond this implementation remains the main paper-level concern.')]
    story=[Paragraph('When graph context changes the classifier',styles['Title2']),
           Paragraph('One-page research decision | 7 September 2026 | Private | Completed evidence, not a new submission draft',styles['Small2']),
           Image(str(root/'figures/class_reference_kv.png'),width=520,height=308.75),
           Image(str(root/'figures/kv_generality.png'),width=540,height=128.25),Spacer(1,5)]
    for title,body in sections:
        story.append(Paragraph(f'<b>{title}.</b> {body}',styles['Body2']))
    story.append(Paragraph('A: pooled AUC; B,D: within-episode AUC. B-C: Hong Kong to political. Shared colors: purple keys, teal values, orange joint replacement. 128 episodes/cell; streams can share accounts and are not training seeds. Evidence: FINDINGS_CLASS_REFERENCE_KV, DECISION_KV_GENERALITY, FINDINGS_PUBLIC_BOUNDARY. Private; reviewer access TBD.',styles['Small2']))
    SimpleDocTemplate(str(out),pagesize=letter,leftMargin=36,rightMargin=36,topMargin=25,bottomMargin=25,
        title='Class-reference construction: contribution decision',author='Private research working draft').build(story)
    print(out)


if __name__=='__main__': main()
