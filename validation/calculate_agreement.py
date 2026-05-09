"""
Calculate Inter-Rater Agreement
Analyzes expert annotations vs automated labels
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

def calculate_agreement(expert_file):
    """
    Calculate agreement metrics between expert and automated labels
    
    Args:
        expert_file: Path to completed Excel file from expert
    """
    
    # Load completed expert review
    df = pd.read_excel(expert_file)
    
    # Clean up data
    df = df[df['Expert_Label'].notna()]  # Remove empty rows
    df = df[df['Expert_Label'] != 'Unclear']  # Remove unclear cases
    
    # Convert labels to binary
    def label_to_binary(label):
        if isinstance(label, str):
            if 'Severe' in label:
                return 1
            elif 'At-Risk' in label:
                return 0
        return label
    
    automated_labels = df['Automated_Label'].apply(label_to_binary).values
    expert_labels = df['Expert_Label'].apply(label_to_binary).values
    
    # Calculate metrics
    print("=" * 70)
    print("INTER-RATER RELIABILITY ANALYSIS")
    print("=" * 70)
    print(f"\nTotal cases reviewed: {len(df)}")
    print(f"Valid cases (excluding 'Unclear'): {len(expert_labels)}")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(expert_labels, automated_labels)
    print(f"\n📊 Cohen's Kappa: {kappa:.3f}")
    
    # Interpret Kappa
    if kappa > 0.80:
        interpretation = "Almost Perfect Agreement"
    elif kappa > 0.60:
        interpretation = "Substantial Agreement"
    elif kappa > 0.40:
        interpretation = "Moderate Agreement"
    elif kappa > 0.20:
        interpretation = "Fair Agreement"
    else:
        interpretation = "Slight Agreement"
    
    print(f"   Interpretation: {interpretation}")
    
    # Overall accuracy
    agreement = np.mean(expert_labels == automated_labels)
    print(f"\n✓ Overall Agreement: {agreement*100:.1f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(expert_labels, automated_labels)
    print(f"\n📋 Confusion Matrix:")
    print(f"                 Automated: At-Risk | Automated: Severe")
    print(f"Expert: At-Risk     {cm[0,0]:5d}       |      {cm[0,1]:5d}")
    print(f"Expert: Severe      {cm[1,0]:5d}       |      {cm[1,1]:5d}")
    
    # Per-class agreement
    class_0_agreement = cm[0,0] / (cm[0,0] + cm[0,1])
    class_1_agreement = cm[1,1] / (cm[1,0] + cm[1,1])
    
    print(f"\n📈 Per-Class Agreement:")
    print(f"   At-Risk (Class 0): {class_0_agreement*100:.1f}%")
    print(f"   Severe (Class 1):  {class_1_agreement*100:.1f}%")
    
    # Classification Report
    print(f"\n📑 Detailed Classification Report:")
    print(classification_report(expert_labels, automated_labels, 
                                target_names=['At-Risk', 'Severe']))
    
    # Confidence analysis
    if 'Expert_Confidence' in df.columns:
        df['Expert_Confidence'] = df['Expert_Confidence'].str.strip()
        print(f"\n🎯 Agreement by Expert Confidence:")
        for conf in ['High', 'Medium', 'Low']:
            conf_df = df[df['Expert_Confidence'] == conf]
            if len(conf_df) > 0:
                conf_auto = conf_df['Automated_Label'].apply(label_to_binary).values
                conf_expert = conf_df['Expert_Label'].apply(label_to_binary).values
                conf_agreement = np.mean(conf_auto == conf_expert)
                print(f"   {conf:6s} confidence: {conf_agreement*100:.1f}% "
                      f"(n={len(conf_df)})")
    
    # Disagreement analysis
    disagreements = df[automated_labels != expert_labels]
    print(f"\n⚠️  Disagreements: {len(disagreements)} cases")
    if len(disagreements) > 0:
        print("\nTop disagreement cases:")
        for idx, row in disagreements.head(5).iterrows():
            print(f"  ID {row['ID']}: Auto={row['Automated_Label']}, "
                  f"Expert={row['Expert_Label']}, "
                  f"Conf={row.get('Confidence', 'N/A')}")
    
    # Save results
    results = {
        'total_cases': len(expert_labels),
        'cohens_kappa': kappa,
        'interpretation': interpretation,
        'overall_agreement': agreement,
        'class_0_agreement': class_0_agreement,
        'class_1_agreement': class_1_agreement,
        'disagreements': len(disagreements)
    }
    
    # Save CSV results
    results_df = pd.DataFrame([results])
    results_df.to_csv('validation_results.csv', index=False)
    print(f"\n✓ Results saved to: validation_results.csv")

    # Write a ready-to-paste text report
    unclear_count = df_orig[df_orig['Expert_Label'].str.strip().str.lower() == 'unclear'].shape[0] \
                    if 'df_orig' in dir() else "?"

    report_lines = [
        "=" * 70,
        "MOODMIRROR — EXPERT VALIDATION AGREEMENT REPORT",
        "=" * 70,
        "",
        f"Cases reviewed:            {len(df)}",
        f"Cases excluded (Unclear):  {unclear_count}",
        f"Valid cases analysed:      {len(expert_labels)}",
        "",
        f"Overall agreement:         {agreement*100:.1f}%",
        f"Cohen's kappa (κ):         {kappa:.3f}",
        f"κ interpretation:          {interpretation}",
        f"At-Risk agreement:         {class_0_agreement*100:.1f}%",
        f"Severe Risk agreement:     {class_1_agreement*100:.1f}%",
        "",
        "Thesis citation:",
        f'  "Dataset labels were independently validated by [Expert Name],',
        f'  [Credentials], through a blinded review of 20 stratified,',
        f'  anonymised real-Reddit user profiles. Inter-rater reliability',
        f'  analysis yielded Cohen\'s κ = {kappa:.3f} ({interpretation}),',
        f'  with {agreement*100:.1f}% overall label concordance."',
        "",
        "=" * 70,
    ]
    report_text = "\n".join(report_lines)
    with open('validation_agreement_report.txt', 'w') as f:
        f.write(report_text)
    print("✓ Text report → validation_agreement_report.txt")
    print("  (copy the stats directly into expert_validation_certificate.md)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import sys

    # Default: look for the 30-min session file; fall back to legacy file
    candidates = [
        "expert_session_20cases.xlsx",
        "expert_review_sample_COMPLETED.xlsx",
    ]

    if len(sys.argv) > 1:
        expert_file = sys.argv[1]
    else:
        expert_file = next((f for f in candidates if __import__('os').path.exists(f)), candidates[0])

    print(f"Looking for completed expert file: {expert_file}")
    print("(Pass a filename as argument if using a different file)\n")

    try:
        results = calculate_agreement(expert_file)

        print("\n📝 Copy these into expert_validation_certificate.md:")
        print(f"   Cohen's κ          : {results['cohens_kappa']:.3f}")
        print(f"   Interpretation     : {results['interpretation']}")
        print(f"   Overall agreement  : {results['overall_agreement']*100:.1f}%")
        print(f"   At-Risk agreement  : {results['class_0_agreement']*100:.1f}%")
        print(f"   Severe agreement   : {results['class_1_agreement']*100:.1f}%")

    except FileNotFoundError:
        print(f"❌ File not found: {expert_file}")
        print("   Complete the expert session first, then run this script.")
