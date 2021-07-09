//Show classification buttons once video has loaded
$(document).ready(function(){
	$.ajaxSetup({ cache: false });
     // or iPhones don't get fresh data
    $('#tablecontainer').on('click','.training-correction', function(){
        $t = $(this);

        file_name = $(this).closest('div').attr('data-filename');
        value = $(this).attr('value');
        
        data_to_send = {
            file_name: file_name,
            value: value,
        };

        $.ajax({
            url: '/correcttrainingdata',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });
        
    });

    $('#pagcontainer').on("click",'.page-item-ajax', function(){
        $t = $(this);
        value = $(this).attr('value');
        
        date = new Date;
    
        $.ajax({
            url: value,
            contentType: "application/json",
            data: JSON.stringify(date),
            type: 'POST',
            success: function(response){
                $('#pagcontainer').html(response['pagination_nav']);
                $('#tablecontainer').html(response['table']);
            },
            error: function(error){
                console.log(error);
            }
        });
        
    });
});

$("#video-feed").on('load', function(){
    var today = new Date();
    $.ajax({
        url: '/dummyajax',
        data: today,
        type: 'POST',
        success: function(response){
            console.log(response);
        },
        error: function(error){
            console.log(error);
        }
    });
    $('.loader').addClass('d-none');
    $('.btn-group-live').removeClass('d-none');
    $('.btn-group-live').addClass('d-block');
    $('.bk').removeClass('d-none');
});

//Send classification data point to be saved from live stream
$(function(){
    $('button.live-classification').click(function(){
        value = $(this).attr('value');
        $.ajax({
            url: '/classify',
            data: value,
            type: 'POST',
            success: function(response){
                $(".alert-container").prepend("<div class='alert alert-success alert-dismissible fade show' role='alert'><strong>Success! </strong>" + response + "<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button></div>");
                $(".alert").delay(3000).slideUp(200, function() {
                    $(this).removeClass('show');
                });
                
                date = new Date;
                $.ajax({
                    url: '/data?page=1&ajax=1',
                    contentType: "application/json",
                    data: JSON.stringify(date),
                    type: 'POST',
                    success: function(response){
                        $('#pagcontainer').html(response['pagination_nav']);
                        $('#tablecontainer').html(response['table']);
                        $('#graphcontainer').html(response['graph_div']);
                        $('#scriptcontainer').html(response['graph_script']);
                    },
                    error: function(error){
                        console.log(error);
                    }
                });
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});

//Send classification data point to be saved
$(function(){
    $('button.correction').click(function(){
        $t = $(this);
        new_value = $(this).attr('value')

        data_to_send = {
            id: $(this).closest('tr').children('td.id-column').text(),
            value: new_value
        };
        $.ajax({
            url: '/correctdatapoint',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                if ( new_value == 'Empty'){
                    old_value = 'Baby';
                } else {
                    old_value = 'Empty';
                }
                $t.val(old_value);
                $t.text(old_value);
                $t.closest('div').siblings('button.dropdown-toggle').text(new_value);

                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});



$('button.reading-classification').click(function(){
    $t = $(this);
    value = $(this).attr('value');
    id = $(this).closest('tr').children('td.id-column').text();
    
    data_to_send = {
        id: id,
        value: value,
    };
    $.ajax({
        url: '/classifyreading',
        contentType: "application/json",
        data: JSON.stringify(data_to_send),
        type: 'POST',
        success: function(response){
            $t.closest('tr').find('div.add-to-training').addClass('d-none');
            $t.closest('tr').find('div.included-in').removeClass('d-none');
            console.log(response);
        },
        error: function(error){
            console.log(error);
        }
    });
    
});

$('#playlivetext').click(function(){
    $('#myModal').modal();
});