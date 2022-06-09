Access to HAICGU
================

Getting an Account
------------------

Currently, accounts can only be created manually. Please send the following information:

- Full Name
- Email Adress
- Public SSH key in OpenSSH format

to haicgu@fias.uni-frankfurt.de

.. attention:: If you want to use the AI nodes/hardware, please also specify this in the email, as you need to be added to a specific user group in order to access the hardware and libraries. You can also ask to be added to this group at a later date.

Accessing the cluster
---------------------

You have to add some clauses to your ssh config::

    # GUOEHI
    Host guoehi
      HostName 141.2.112.17
      User yourusername
      ServerAliveInterval 60
      IdentityFile /path/to/your/private/ssh/key
      IdentitiesOnly yes
    Host guoehi-dev
      HostName dev
      user yourusername
      ProxyCommand ssh -q -W %h:%p guoehi

Replacing ``yourusername`` and ``/path/to/your/private/ssh/key`` accordingly.

On a MacOS or Linux machine, this goes into ``~/.ssh/config`` .

On Windows 10 and higher, when using the built-in OpenSSH, this goes into ``%USERPROFILE%\.ssh\config`` (not tested)

You can now access the dev node by typing::

    ssh guoehi-dev

on your machine

.. note:: The old "guoehi" name is still used by multiple internal software components, therefore it has been left in the ssh config to avoid confusing users if they see it in the output of a program on the cluster.

