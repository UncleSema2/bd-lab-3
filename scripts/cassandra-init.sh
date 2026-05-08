#!/bin/bash
set -e

ROLE="${CASSANDRA_USERNAME}"
PASSWORD="${CASSANDRA_PASSWORD}"
KEYSPACE="${CASSANDRA_KEYSPACE}"

docker-entrypoint.sh cassandra &

for i in $(seq 1 60); do
  if cqlsh -u "${ROLE}" -p "${PASSWORD}" -e 'describe cluster' &>/dev/null; then
    echo "Cassandra is ready"
    break
  fi
  echo "Waiting for Cassandra to be ready... ($i/60)"
  sleep 5
done

cqlsh -u "${ROLE}" -p "${PASSWORD}" -e "CREATE KEYSPACE IF NOT EXISTS ${KEYSPACE} WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}" || true

cat <<EOF | cqlsh -u "${ROLE}" -p "${PASSWORD}"
USE ${KEYSPACE};

CREATE TABLE IF NOT EXISTS train_data (
    id int PRIMARY KEY,
    mean_radius double,
    mean_texture double,
    mean_perimeter double,
    mean_area double,
    mean_smoothness double,
    mean_compactness double,
    mean_concavity double,
    mean_concave_points double,
    mean_symmetry double,
    mean_fractal_dimension double,
    radius_error double,
    texture_error double,
    perimeter_error double,
    area_error double,
    smoothness_error double,
    compactness_error double,
    concavity_error double,
    concave_points_error double,
    symmetry_error double,
    fractal_dimension_error double,
    worst_radius double,
    worst_texture double,
    worst_perimeter double,
    worst_area double,
    worst_smoothness double,
    worst_compactness double,
    worst_concavity double,
    worst_concave_points double,
    worst_symmetry double,
    worst_fractal_dimension double,
    target int
);

CREATE TABLE IF NOT EXISTS eval_data (
    id int PRIMARY KEY,
    mean_radius double,
    mean_texture double,
    mean_perimeter double,
    mean_area double,
    mean_smoothness double,
    mean_compactness double,
    mean_concavity double,
    mean_concave_points double,
    mean_symmetry double,
    mean_fractal_dimension double,
    radius_error double,
    texture_error double,
    perimeter_error double,
    area_error double,
    smoothness_error double,
    compactness_error double,
    concavity_error double,
    concave_points_error double,
    symmetry_error double,
    fractal_dimension_error double,
    worst_radius double,
    worst_texture double,
    worst_perimeter double,
    worst_area double,
    worst_smoothness double,
    worst_compactness double,
    worst_concavity double,
    worst_concave_points double,
    worst_symmetry double,
    worst_fractal_dimension double,
    target int
);
EOF

echo "Loading train data..."
paste -d',' /data/processed/X_train.csv /data/processed/y_train.csv | tail -n +2 | awk -F',' '{print $1","$3","$4","$5","$6","$7","$8","$9","$10","$11","$12","$13","$14","$15","$16","$17","$18","$19","$20","$21","$22","$23","$24","$25","$26","$27","$28","$29","$30","$31","$32","$33}' > /tmp/load_train.csv
cqlsh -u "${ROLE}" -p "${PASSWORD}" -e "USE ${KEYSPACE}; COPY train_data (id, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension, target) FROM '/tmp/load_train.csv';"

echo "Loading eval data..."
paste -d',' /data/processed/X_test.csv /data/processed/y_test.csv | tail -n +2 | awk -F',' '{print $1","$3","$4","$5","$6","$7","$8","$9","$10","$11","$12","$13","$14","$15","$16","$17","$18","$19","$20","$21","$22","$23","$24","$25","$26","$27","$28","$29","$30","$31","$32","$33}' > /tmp/load_eval.csv
cqlsh -u "${ROLE}" -p "${PASSWORD}" -e "USE ${KEYSPACE}; COPY eval_data (id, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension, target) FROM '/tmp/load_eval.csv';"

echo "Data loading complete"

tail -f /dev/null
