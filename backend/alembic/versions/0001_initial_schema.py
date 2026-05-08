"""Initial enterprise platform schema."""

from alembic import op
import sqlalchemy as sa


revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


user_role = sa.Enum('ADMIN', 'RESEARCHER', 'DOCTOR', 'LAB_TECHNICIAN', name='userrole')
prediction_label = sa.Enum('RESISTANT', 'SUSCEPTIBLE', name='predictionlabel')
report_status = sa.Enum('GENERATED', 'DELIVERED', 'FAILED', name='reportstatus')


def upgrade() -> None:
    user_role.create(op.get_bind(), checkfirst=True)
    prediction_label.create(op.get_bind(), checkfirst=True)
    report_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'users',
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', user_role, nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    op.create_table(
        'microbes',
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('taxonomy_group', sa.String(length=120), nullable=True),
        sa.Column('genome_reference', sa.String(length=255), nullable=True),
        sa.Column('baseline_resistance_rate', sa.Float(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_microbes_name'), 'microbes', ['name'], unique=True)

    op.create_table(
        'antibiotics',
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('antibiotic_class', sa.String(length=120), nullable=True),
        sa.Column('who_awarea_category', sa.String(length=50), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_antibiotics_name'), 'antibiotics', ['name'], unique=True)

    op.create_table(
        'patients',
        sa.Column('external_id', sa.String(length=100), nullable=False),
        sa.Column('first_name', sa.String(length=120), nullable=False),
        sa.Column('last_name', sa.String(length=120), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=False),
        sa.Column('sex', sa.String(length=20), nullable=False),
        sa.Column('facility_name', sa.String(length=255), nullable=False),
        sa.Column('created_by_id', sa.String(length=36), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_patients_external_id'), 'patients', ['external_id'], unique=True)

    op.create_table(
        'predictions',
        sa.Column('patient_id', sa.String(length=36), nullable=True),
        sa.Column('microbe_id', sa.String(length=36), nullable=False),
        sa.Column('antibiotic_id', sa.String(length=36), nullable=False),
        sa.Column('requested_by_id', sa.String(length=36), nullable=False),
        sa.Column('prediction_label', prediction_label, nullable=False),
        sa.Column('resistant_probability', sa.Float(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('explanation_text', sa.Text(), nullable=False),
        sa.Column('recommended_alternatives', sa.JSON(), nullable=False),
        sa.Column('shap_summary', sa.JSON(), nullable=False),
        sa.Column('model_version', sa.String(length=100), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['antibiotic_id'], ['antibiotics.id']),
        sa.ForeignKeyConstraint(['microbe_id'], ['microbes.id']),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id']),
        sa.ForeignKeyConstraint(['requested_by_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'resistance_history',
        sa.Column('microbe_id', sa.String(length=36), nullable=False),
        sa.Column('antibiotic_id', sa.String(length=36), nullable=False),
        sa.Column('region', sa.String(length=120), nullable=True),
        sa.Column('facility_name', sa.String(length=255), nullable=True),
        sa.Column('sample_date', sa.Date(), nullable=True),
        sa.Column('result_resistant', sa.Boolean(), nullable=False),
        sa.Column('mic_value', sa.Float(), nullable=True),
        sa.Column('genomic_markers', sa.JSON(), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['antibiotic_id'], ['antibiotics.id']),
        sa.ForeignKeyConstraint(['microbe_id'], ['microbes.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'reports',
        sa.Column('prediction_id', sa.String(length=36), nullable=False),
        sa.Column('created_by_id', sa.String(length=36), nullable=False),
        sa.Column('file_path', sa.String(length=500), nullable=False),
        sa.Column('status', report_status, nullable=False),
        sa.Column('emailed_to', sa.String(length=255), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id']),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'uploaded_datasets',
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('source', sa.String(length=120), nullable=False),
        sa.Column('schema_version', sa.String(length=50), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('uploaded_by_id', sa.String(length=36), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['uploaded_by_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'model_metadata',
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('version', sa.String(length=100), nullable=False),
        sa.Column('algorithm', sa.String(length=120), nullable=False),
        sa.Column('metrics_json', sa.JSON(), nullable=False),
        sa.Column('artifact_uri', sa.String(length=500), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'analytics_snapshots',
        sa.Column('metric_name', sa.String(length=120), nullable=False),
        sa.Column('time_bucket', sa.String(length=50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('dimensions_json', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_analytics_snapshots_metric_name'), 'analytics_snapshots', ['metric_name'], unique=False)

    op.create_table(
        'audit_logs',
        sa.Column('actor_id', sa.String(length=36), nullable=True),
        sa.Column('action', sa.String(length=120), nullable=False),
        sa.Column('entity_type', sa.String(length=120), nullable=False),
        sa.Column('entity_id', sa.String(length=120), nullable=True),
        sa.Column('details_json', sa.JSON(), nullable=False),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['actor_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_table('audit_logs')
    op.drop_index(op.f('ix_analytics_snapshots_metric_name'), table_name='analytics_snapshots')
    op.drop_table('analytics_snapshots')
    op.drop_table('model_metadata')
    op.drop_table('uploaded_datasets')
    op.drop_table('reports')
    op.drop_table('resistance_history')
    op.drop_table('predictions')
    op.drop_index(op.f('ix_patients_external_id'), table_name='patients')
    op.drop_table('patients')
    op.drop_index(op.f('ix_antibiotics_name'), table_name='antibiotics')
    op.drop_table('antibiotics')
    op.drop_index(op.f('ix_microbes_name'), table_name='microbes')
    op.drop_table('microbes')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')

    report_status.drop(op.get_bind(), checkfirst=True)
    prediction_label.drop(op.get_bind(), checkfirst=True)
    user_role.drop(op.get_bind(), checkfirst=True)
