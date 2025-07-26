#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _generic_muscle_iaf_cell_reg(void);
extern void _generic_neuron_iaf_cell_reg(void);
extern void _neuron_to_muscle_exc_syn_reg(void);
extern void _neuron_to_muscle_inh_syn_reg(void);
extern void _neuron_to_neuron_elec_syn_reg(void);
extern void _neuron_to_neuron_exc_syn_reg(void);
extern void _neuron_to_neuron_inh_syn_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"generic_muscle_iaf_cell.mod\"");
    fprintf(stderr, " \"generic_neuron_iaf_cell.mod\"");
    fprintf(stderr, " \"neuron_to_muscle_exc_syn.mod\"");
    fprintf(stderr, " \"neuron_to_muscle_inh_syn.mod\"");
    fprintf(stderr, " \"neuron_to_neuron_elec_syn.mod\"");
    fprintf(stderr, " \"neuron_to_neuron_exc_syn.mod\"");
    fprintf(stderr, " \"neuron_to_neuron_inh_syn.mod\"");
    fprintf(stderr, "\n");
  }
  _generic_muscle_iaf_cell_reg();
  _generic_neuron_iaf_cell_reg();
  _neuron_to_muscle_exc_syn_reg();
  _neuron_to_muscle_inh_syn_reg();
  _neuron_to_neuron_elec_syn_reg();
  _neuron_to_neuron_exc_syn_reg();
  _neuron_to_neuron_inh_syn_reg();
}

#if defined(__cplusplus)
}
#endif
