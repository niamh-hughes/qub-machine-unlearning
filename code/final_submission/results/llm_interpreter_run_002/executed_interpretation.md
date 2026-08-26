### Executed interpretation

- Few-shot Qwen changed supported-policy accuracy by +40.0 percentage points relative to rules and +15.0 points relative to zero-shot Qwen.
- For the selected LLM condition, the easiest policy was `recipient_withdrawal` (100.0%) and the hardest was `donor_withdrawal` (93.8%).
- The lowest difficulty-group accuracy for the selected LLM was `straightforward` (98.3%).
- Entity extraction, policy classification, and row-set overlap are reported separately because one can be correct while another fails.
- Unsafe execution is more serious than conservative review: `unknown` delays action, whereas an incorrect executable policy can under-delete or over-delete training data.

These observations describe this fixed benchmark. They do not establish causation, regulatory compliance, privacy, erasure, or safety for autonomous deployment.