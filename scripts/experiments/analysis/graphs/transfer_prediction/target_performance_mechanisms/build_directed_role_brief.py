"""Private one-page decision brief. Not a manuscript expansion."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor


def main():
    root=Path(__file__).parent
    repo=next(p for p in root.parents if (p/'AGENTS.md').is_file())
    out=repo/'output/pdf/directed_context_decision.pdf'
    out.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BriefTitle',fontName='Helvetica-Bold',fontSize=16,leading=19,spaceAfter=5,textColor=HexColor('#173b55')))
    styles.add(ParagraphStyle(name='BriefBody',fontName='Helvetica',fontSize=9.3,leading=12,spaceAfter=7))
    styles.add(ParagraphStyle(name='BriefSmall',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=6,textColor=HexColor('#555555')))
    story=[Paragraph('Directed context changes how supports and queries agree',styles['BriefTitle']),
           Paragraph('Research decision | 7 September 2026 | Private | Complete fixed-checkpoint test; no new training',styles['BriefSmall']),
           Image(str(root/'figures/directed_role_argument.png'),width=540,height=198),Spacer(1,7)]
    paragraphs=[
        '<b>Explanation supported.</b> Early predictions depend on the relative orientation of support and query contexts, not only on which biographies are available. With sampled nodes, features and the undirected graph fixed, reversing supports alone changes mean episode AUC from .692/.685 to .413/.408; reversing queries alone gives .428/.400. Reversing both partially restores ranking to .581/.581. Both-sided reversal exceeds either one-sided intervention for all nine sources in both streams. Support-only reversal leaves final query vectors bit-exact, so its effect reaches decisions through the constructed class references.',
        '<b>The decisive counterexample.</b> The full predeclared prediction <b>fails</b>: joint reversal retains only 42.5%/44.0% of the intact above-chance advantage, below the required 50% (short dotted marks in A). This is partial recovery, not orientation-invariant inference or a repaired classifier. The proposed signed-degree-matching bridge to the natural training decline is not established. We retain this failure rather than substitute a better-performing intermediate readout.',
        '<b>What training changes.</b> The orientation-alignment contrast C falls from 21.59/22.89 to 6.34/5.89 AUC points, decreasing for every source in both streams. Pooled-feature ridge retains a contrast near 15 to 14 points; the post-projection U1 probe falls from roughly 27 to 19. These are supporting stage comparisons, not an additive localization. Much of the native contraction combines worse aligned performance with one-sided conditions moving toward chance; reduced sensitivity is not evidence of useful robustness.',
        '<b>Generality and strongest alternative.</b> The result spans nine source checkpoints, but only one target population, one architecture and one training seed per source. Streams are not seeds. Reversal preserves membership and edge counts, improving on the earlier retweet-to-follow comparison, yet changes which feature messages reach each node. Generic support/query distribution mismatch remains a competing explanation; the intervention does not isolate degree. The prior follow result chiefly removed an early advantage and remains boundary evidence, not a repair.',
        '<b>Exact contribution and decision.</b> A controlled, source-broad signature of support-query orientation dependence that training attenuates despite persistent pooled-readout sensitivity. This goes beyond isolated edge-removal effects, but not to an identified signed-role algorithm, a support-only defect, or a deployment method. Keep this signature as a constraint on the explanation; reject the stronger bridge in its prespecified form. Do not expand the manuscript or launch a wider reversal sweep on this result alone. The intended strong mechanism contribution remains unresolved.'
    ]
    story.extend(Paragraph(t,styles['BriefBody']) for t in paragraphs)
    story.append(Paragraph('Evidence: directed_roles_20260907.json and directed_roles_summary_20260907.json; code 665469c2. All 36 checkpoint/stream blocks, 432 metric rows, 1,152 native-batch parity checks and 2,304 query-role checks completed. Activations remain private on Tucker.',styles['BriefSmall']))
    SimpleDocTemplate(str(out),pagesize=letter,leftMargin=36,rightMargin=36,topMargin=27,bottomMargin=25,
                      title='Directed context: research decision',author='Research working draft').build(story)
    print(out)


if __name__=='__main__': main()
