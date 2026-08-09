// Delete existing nodes and relationships
MATCH (n) DETACH DELETE n;

// Give each node a unique code name
CREATE CONSTRAINT lpbf_code_name_unique IF NOT EXISTS FOR (n:LPBF_Node) REQUIRE n.code_name IS UNIQUE;

// Create nodes
UNWIND ['material_type','material_form','material_diameter','laser_power','hatching_distance','layer_thickness','rotation_angle','scanning_speed','elongation_to_failure','tensile_strength','yield_stress','hardness'] AS n
CREATE (:LPBF_Node {code_name:n});

// Define relationships
UNWIND [
  ['material_type','laser_power'],['material_type','hatching_distance'],['material_type','layer_thickness'],['material_type','rotation_angle'],['material_type','scanning_speed'],
  ['material_form','material_diameter'],['material_form','laser_power'],['material_form','hatching_distance'],['material_form','layer_thickness'],['material_form','rotation_angle'],['material_form','scanning_speed']
] AS e

// Find and create material-parameter relationships
MATCH (a:LPBF_Node {code_name:e[0]}),(b:LPBF_Node {code_name:e[1]})
CREATE (a)-[:DIRECT_INFLUENCE]->(b);

// Assign numerical process parameters as a source node
UNWIND ['material_diameter','laser_power','hatching_distance','layer_thickness','rotation_angle','scanning_speed'] AS s

// Assign targets as target node
UNWIND ['elongation_to_failure','tensile_strength','yield_stress','hardness'] AS t

// Find and create parameter-target relationships
MATCH (a:LPBF_Node {code_name:s}),(b:LPBF_Node {code_name:t})
CREATE (a)-[:DIRECT_INFLUENCE]->(b);

// Find and display direct influence relationships in completed graph
MATCH p=(:LPBF_Node)-[:DIRECT_INFLUENCE]->(:LPBF_Node)
RETURN p;