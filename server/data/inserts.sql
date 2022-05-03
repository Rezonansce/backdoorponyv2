INSERT INTO actions (name, info, link, is_attack)
                      VALUES ('BADNETS', 'Some info on the badnets attack...', 'https://arxiv.org/pdf/1708.06733.pdf', true),
                             ('STRIP', 'Some info on the STRIP defence...', 'https://arxiv.org/pdf/1902.06531.pdf', false);
INSERT INTO params (name, info, lower_bound, upper_bound, type)
						VALUES ('Number of epochs', 'Some info on the parameter...', 0.0, 100.0, 'images');
INSERT INTO defaults (action_id, param_id, value)
                        VALUES ('BADNETS', 'Number of epochs', 5);
