// Delete existing nodes and relationships
MATCH (n) DETACH DELETE n;

// Give each node a unique code name
CREATE CONSTRAINT lpbf_code_name_unique IF NOT EXISTS FOR (n:LPBF_Node) REQUIRE n.code_name IS UNIQUE;

// Create nodes
UNWIND ['material_type','material_form','material_diameter','laser_power','hatching_distance','layer_thickness','rotation_angle','scanning_speed','anisotropy','porosity_defects','microstructure','melt_pool_behaviour','energy_input','thermal_gradient','elongation_to_failure','tensile_strength','yield_stress','hardness'] AS code
MERGE (:LPBF_Node {code_name:code});

// Assign materials as source node
UNWIND ['material_type','material_form'] AS s

// Assign input parameters as target node
UNWIND ['material_diameter','laser_power','hatching_distance','layer_thickness','rotation_angle','scanning_speed'] AS t

// Find and create material-parameter relationships
MATCH (a:LPBF_Node {code_name:s}), (b:LPBF_Node {code_name:t})
MERGE (a)-[r:DIRECT_INFLUENCE]->(b) SET r.weight=1.0;

// Assign input parameters as source node
UNWIND ['material_diameter','laser_power','hatching_distance','layer_thickness','rotation_angle','scanning_speed'] AS s

// Assign physical mechanisms as target node
UNWIND ['anisotropy','porosity_defects','microstructure','melt_pool_behaviour','energy_input','thermal_gradient'] AS t

// Find and create parameter-physicalmechanism relationships
MATCH (a:LPBF_Node {code_name:s}), (b:LPBF_Node {code_name:t})
MERGE (a)-[r:DIRECT_INFLUENCE]->(b) SET r.weight=1.0;

// Find and create physical mechanism-target relationships
UNWIND [
  ['anisotropy',[['hardness',.50],['yield_stress',.75],['tensile_strength',.75],['elongation_to_failure',1.00]]],
  ['energy_input',[['hardness',.75],['yield_stress',.75],['tensile_strength',.75],['elongation_to_failure',.50]]],
  ['melt_pool_behaviour',[['hardness',.50],['yield_stress',.75],['tensile_strength',1.00],['elongation_to_failure',.75]]],
  ['porosity_defects',[['hardness',.50],['yield_stress',.75],['tensile_strength',1.00],['elongation_to_failure',1.00]]],
  ['microstructure',[['hardness',1.00],['yield_stress',1.00],['tensile_strength',.75],['elongation_to_failure',.75]]],
  ['thermal_gradient',[['hardness',.75],['yield_stress',.75],['tensile_strength',.50],['elongation_to_failure',.50]]]
] AS x
UNWIND x[1] AS y
MATCH (a:LPBF_Node {code_name:x[0]}), (b:LPBF_Node {code_name:y[0]})
MERGE (a)-[r:DIRECT_INFLUENCE]->(b) SET r.weight=y[1];

// Find and display direct influence relationships in completed graph
MATCH (s:LPBF_Node)-[r:DIRECT_INFLUENCE]->(t:LPBF_Node)
RETURN s.code_name AS source, t.code_name AS target, r.weight
ORDER BY source, target;
