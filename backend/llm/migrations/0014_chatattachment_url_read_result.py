from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("llm", "0013_merge_attachment_and_legacy_usage")]

    operations = [
        migrations.AddField(
            model_name="chatattachment",
            name="url_read_result",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
